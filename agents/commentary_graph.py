"""
Multilingual Multi-Agent Commentary and Q&A Engine built with LangGraph and LangChain.
Implements StateGraph orchestration with Lead Commentator, Color Analyst, Guardrail Reflection Node,
and Q&A Expert Agent across English, Tamil (தமிழ்), and Hindi (हिन्दी).
"""

import json
import os
import time
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.config import AgentConfig
from agents.guardrails import CommentaryGuardrails
from agents.tools import CricketStatsTools
from monitoring.metrics import MetricsManager

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# =====================================================================
# 1. LangGraph State Schemas
# =====================================================================


class CommentaryState(TypedDict, total=False):
    delivery: dict[str, Any]
    match_state: dict[str, Any]
    language: str
    lead_commentary: str
    analyst_commentary: str | None
    lead_persona: str
    analyst_persona: str | None
    guardrail_passed: bool
    guardrail_reason: str
    reflection_attempts: int
    max_reflection_retries: int
    correction_critique: str
    error: str | None


class QAState(TypedDict, total=False):
    question: str
    match_state: dict[str, Any]
    language: str
    session_deliveries: list[dict[str, Any]]
    match_summary: dict[str, Any]
    answer: str
    grounded_state: dict[str, Any]
    latency_sec: float


# =====================================================================
# 2. Commentary & Q&A Multi-Agent Graph Engine
# =====================================================================


class CommentaryAgentGraph:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or AgentConfig.GROQ_API_KEY
        self.tools = CricketStatsTools()
        self.guardrails = CommentaryGuardrails()
        self.client = None
        self._model_rotation_idx = 0

        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"[AgentGraph] Warning initializing Groq client: {e}")

        # Initialize MLflow GenAI Tracing & Experiment tracking (Fast Non-Blocking)
        if MLFLOW_AVAILABLE:
            try:
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                local_db = os.path.join(base_dir, "data", "mlflow", "mlflow.db")
                fallback_uri = f"sqlite:///{local_db}"
                
                target_uri = os.getenv("MLFLOW_TRACKING_URI", fallback_uri)
                if target_uri.startswith("http"):
                    # Fast socket check (0.15s) to prevent blocking Streamlit startup if server is down
                    from urllib.parse import urlparse
                    import socket
                    parsed = urlparse(target_uri)
                    sock = socket.create_connection((parsed.hostname or "localhost", parsed.port or 5000), timeout=0.15)
                    sock.close()
                    mlflow.set_tracking_uri(target_uri)
                else:
                    mlflow.set_tracking_uri(fallback_uri)
                
                mlflow.set_experiment(AgentConfig.MLFLOW_EXPERIMENT_NAME)
                mlflow.langchain.autolog(log_models=False)
            except Exception:
                try:
                    local_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "mlflow", "mlflow.db"))
                    mlflow.set_tracking_uri(f"sqlite:///{local_db}")
                    mlflow.set_experiment(AgentConfig.MLFLOW_EXPERIMENT_NAME)
                except Exception:
                    pass

        # Build and compile LangGraph workflows
        self.commentary_graph = self._build_commentary_graph()
        self.qa_graph = self._build_qa_graph()

    def _normalize_lang(self, lang: str | None) -> str:
        if not lang:
            return "en"
        l = lang.lower().strip()
        if "ta" in l or "tamil" in l or "தமிழ்" in l:
            return "ta"
        if "hi" in l or "hindi" in l or "हिन्दी" in l or "हिंदी" in l:
            return "hi"
        return "en"

    def _get_next_model(self) -> str:
        """Rotates models round-robin across active models in MODEL_POOL."""
        pool = AgentConfig.MODEL_POOL
        selected = pool[self._model_rotation_idx % len(pool)]
        self._model_rotation_idx += 1
        return selected

    def _call_groq_messages(
        self,
        messages: list[SystemMessage | HumanMessage],
        model: str | None = None,
        lang: str = "en",
    ) -> str:
        """Executes LangChain messages with model pool failover."""
        if not self.client:
            raise RuntimeError(
                "[AgentGraph] Groq client is not initialized. Please configure GROQ_API_KEY."
            )

        primary_candidate = model or self._get_next_model()
        models_to_try = [primary_candidate] + [
            m for m in AgentConfig.MODEL_POOL if m != primary_candidate
        ]

        formatted_msgs = []
        for msg in messages:
            role = "system" if isinstance(msg, SystemMessage) else "user"
            formatted_msgs.append({"role": role, "content": msg.content})

        last_error = None
        for m_name in models_to_try:
            try:
                response = self.client.chat.completions.create(
                    model=m_name,
                    messages=formatted_msgs,
                    temperature=AgentConfig.TEMPERATURE,
                    max_tokens=AgentConfig.MAX_TOKENS,
                )
                text = response.choices[0].message.content
                if text and text.strip():
                    return text.strip()
            except Exception as e:
                last_error = e
                next_idx = models_to_try.index(m_name) + 1
                next_m = (
                    models_to_try[next_idx]
                    if next_idx < len(models_to_try)
                    else "exhausted"
                )
                MetricsManager.record_model_failover(from_model=m_name, to_model=next_m)
                print(
                    f"[AgentGraph] Failover notice from {m_name}: {e}. Trying next model in pool..."
                )
                continue

        # If all models in pool failed, provide graceful offline fallback
        print(
            f"[AgentGraph] Notice: Model pool unreachable ({last_error}). Using resilient fallback."
        )
        user_content = formatted_msgs[-1]["content"] if formatted_msgs else ""
        if "VIEWER QUESTION" in user_content:
            if lang == "ta":
                return "தற்போதைய ஆட்டத்தின் நிலவரப்படி பேட்டிங் அணி சிறப்பாக விளையாடி வருகிறது."
            elif lang == "hi":
                return "वर्तमान मैच की स्थिति के अनुसार बल्लेबाजी टीम मजबूत स्थिति में है।"
            return "Based on the live match state, the batting team is currently in action."

        if lang == "ta":
            return "பந்தவீச்சாளர் பந்தை வீசுகிறார், பேட்ஸ்மேன் கவனமாக எதிர்கொள்கிறார்."
        elif lang == "hi":
            return "गेंदबाज ने गेंद डाली, बल्लेबाज ने समझदारी से शॉट खेला।"
        return "The bowler delivers and the batter works it into the outfield."

    # =====================================================================
    # 3. LangGraph Node Definitions
    # =====================================================================

    def _lead_commentator_node(self, state: CommentaryState) -> dict[str, Any]:
        """LangGraph Node: Lead Play-by-Play Commentator Agent."""
        delivery = state.get("delivery", {})
        match_state = state.get("match_state", {})
        lang_key = self._normalize_lang(state.get("language", "en"))
        critique = state.get("correction_critique", "")

        personas = AgentConfig.PERSONAS.get(lang_key, AgentConfig.PERSONAS["en"])
        lead_persona = personas["lead"]

        system_msg = SystemMessage(
            content=(
                f"You are {lead_persona['name']}, the Lead Play-by-Play Cricket Commentator. "
                f"{lead_persona['style']}\n"
                "STRICT RULES:\n"
                "1. You MUST strictly describe the exact ball event (batter, bowler, runs scored, wickets, extras).\n"
                "2. NEVER invent sixes, fours, or wickets that did not happen on this delivery.\n"
                f"3. Generate output in the target language ({lang_key.upper()}). "
                "Use natural cricket broadcast style without robotic translation.\n"
                "4. Keep output to 1-2 punchy, energetic sentences without commentator quotes or prefixes."
            )
        )

        user_content = (
            f"LIVE MATCH EVENT:\n"
            f"- Over {delivery.get('over')}.{delivery.get('ball')}\n"
            f"- Inning: {delivery.get('inning_num')} ({match_state.get('batting_team')} vs {match_state.get('bowling_team')})\n"
            f"- Bowler: {delivery.get('bowler')}\n"
            f"- Striker: {delivery.get('batter')}\n"
            f"- Non-Striker: {delivery.get('non_striker')}\n"
            f"- Runs Scored: {delivery.get('runs_total')} (Batter: {delivery.get('runs_batter')}, Extras: {delivery.get('runs_extras')})\n"
            f"- Extra Type: {delivery.get('extra_type')}\n"
            f"- Wicket: {delivery.get('is_wicket')} (Out: {delivery.get('player_out')}, Kind: {delivery.get('wicket_kind')}, Fielders: {delivery.get('fielders')})\n"
            f"- Current Score: {match_state.get('batting_team')} {match_state.get('score')}/{match_state.get('wickets')} ({match_state.get('overs_display')} ov)"
        )
        if critique:
            user_content += critique

        human_msg = HumanMessage(content=user_content)

        t_start = time.time()
        commentary = self._call_groq_messages(
            [system_msg, human_msg], model=AgentConfig.FAST_MODEL, lang=lang_key
        )
        MetricsManager.record_commentary_latency(
            "lead_commentator", time.time() - t_start
        )

        return {
            "lead_commentary": commentary,
            "lead_persona": lead_persona["name"],
        }

    def _guardrail_reflection_node(self, state: CommentaryState) -> dict[str, Any]:
        """LangGraph Node: Fact-Grounding Guardrail & Reflection Critic."""
        delivery = state.get("delivery", {})
        commentary = state.get("lead_commentary", "")
        attempts = state.get("reflection_attempts", 0)

        is_valid, reason = self.guardrails.validate_delivery_commentary(
            commentary, delivery
        )

        if is_valid:
            return {
                "guardrail_passed": True,
                "guardrail_reason": "Passed",
                "correction_critique": "",
            }

        MetricsManager.record_guardrail_violation()
        print(
            f"[LangGraph Guardrail Node - Attempt {attempts + 1}] {reason}. Triggering reflection loop..."
        )

        critique = (
            f"\n\n[CORRECTION FEEDBACK - REFLECTION NODE]:\n"
            f"Your previous commentary failed factual grounding: {reason}.\n"
            f"Rewrite the commentary immediately ensuring strict adherence to ground truth:\n"
            f"- Real runs scored on this ball: {delivery.get('runs_total')}\n"
            f"- Is Wicket: {delivery.get('is_wicket')}\n"
            f"Do not claim any boundary or dismissal that did not occur."
        )

        return {
            "guardrail_passed": False,
            "guardrail_reason": reason,
            "reflection_attempts": attempts + 1,
            "correction_critique": critique,
        }

    def _color_analyst_node(self, state: CommentaryState) -> dict[str, Any]:
        """LangGraph Node: Tactical Color Analyst Agent."""
        delivery = state.get("delivery", {})
        match_state = state.get("match_state", {})
        lang_key = self._normalize_lang(state.get("language", "en"))

        is_key_moment = (
            delivery.get("is_wicket")
            or delivery.get("runs_batter", 0) >= 4
            or delivery.get("ball") in [1, 6]
        )
        if not is_key_moment:
            return {"analyst_commentary": None, "analyst_persona": None}

        personas = AgentConfig.PERSONAS.get(lang_key, AgentConfig.PERSONAS["en"])
        analyst_persona = personas["analyst"]

        system_msg = SystemMessage(
            content=(
                f"You are {analyst_persona['name']}, the Expert Cricket Color Analyst. "
                f"{analyst_persona['style']}\n"
                "STRICT RULES:\n"
                "1. Ground your tactical insight in the match situation, strike rates, economy, and bowler tactics.\n"
                f"2. Generate output strictly in the target language ({lang_key.upper()}).\n"
                "3. Keep output to 1-2 analytical, insightful sentences without quotes or commentator prefixes."
            )
        )

        striker = delivery.get("batter", "")
        bowler = delivery.get("bowler", "")
        striker_stats = match_state.get("batters", {}).get(striker, {})
        bowler_stats = match_state.get("bowlers", {}).get(bowler, {})

        user_content = (
            f"TACTICAL MATCH SITUATION:\n"
            f"- Match State: {match_state.get('batting_team')} {match_state.get('score')}/{match_state.get('wickets')} in {match_state.get('overs_display')} overs (CRR: {match_state.get('crr')})\n"
            f"- Target: {match_state.get('target')} (RRR: {match_state.get('rrr')})\n"
            f"- Striker {striker}: {striker_stats.get('runs', 0)} runs off {striker_stats.get('balls', 0)} balls (4s: {striker_stats.get('fours', 0)}, 6s: {striker_stats.get('sixes', 0)})\n"
            f"- Bowler {bowler}: {bowler_stats.get('runs', 0)} runs conceded, {bowler_stats.get('wickets', 0)} wickets\n"
            f"- Just Happened: {striker} facing {bowler}, resulting in {delivery.get('runs_total')} runs (Wicket: {delivery.get('is_wicket')})"
        )
        human_msg = HumanMessage(content=user_content)

        t_start = time.time()
        analyst_text = self._call_groq_messages(
            [system_msg, human_msg],
            model=AgentConfig.PRIMARY_MODEL,
            lang=lang_key,
        )
        MetricsManager.record_commentary_latency("color_analyst", time.time() - t_start)

        return {
            "analyst_commentary": analyst_text,
            "analyst_persona": analyst_persona["name"],
        }

    # =====================================================================
    # 4. LangGraph Graph Builder & Router
    # =====================================================================

    def _should_reflect(self, state: CommentaryState) -> str:
        """Conditional Router Edge: Determines if guardrail reflection retry is needed."""
        passed = state.get("guardrail_passed", True)
        attempts = state.get("reflection_attempts", 0)
        max_retries = state.get("max_reflection_retries", 2)

        if not passed and attempts < max_retries:
            return "reflect"
        return "continue"

    def _build_commentary_graph(self):
        """Constructs and compiles the StateGraph for Ball-by-Ball Commentary."""
        workflow = StateGraph(CommentaryState)

        # Register Nodes
        workflow.add_node("lead_commentator", self._lead_commentator_node)
        workflow.add_node("guardrail_checker", self._guardrail_reflection_node)
        workflow.add_node("color_analyst", self._color_analyst_node)

        # Register Edges
        workflow.add_edge(START, "lead_commentator")
        workflow.add_edge("lead_commentator", "guardrail_checker")

        # Conditional Edge for Reflection Loop
        workflow.add_conditional_edges(
            "guardrail_checker",
            self._should_reflect,
            {
                "reflect": "lead_commentator",
                "continue": "color_analyst",
            },
        )

        workflow.add_edge("color_analyst", END)
        return workflow.compile()

    def _build_qa_graph(self):
        """Constructs and compiles the StateGraph for Viewer Match Q&A."""

        def _retrieve_context(state: QAState) -> dict[str, Any]:
            summary = self.tools.get_current_match_summary()
            return {"match_summary": summary}

        def _qa_agent_node(state: QAState) -> dict[str, Any]:
            question = state.get("question", "")
            match_state = state.get("match_state", {})
            lang_key = self._normalize_lang(state.get("language", "en"))
            session_deliveries = state.get("session_deliveries", [])
            match_summary = state.get("match_summary", {})

            if any("\u0b80" <= c <= "\u0bff" for c in question):
                lang_key = "ta"
            elif any("\u0900" <= c <= "\u097f" for c in question):
                lang_key = "hi"

            recent_events = []
            if session_deliveries:
                for d in session_deliveries[-6:]:
                    recent_events.append(
                        f"Over {d.get('over')}.{d.get('ball')}: {d.get('batter')} vs {d.get('bowler')} -> {d.get('runs', d.get('runs_total', 0))} runs"
                        + (
                            f" (WICKET: {d.get('player_out', 'Out')})"
                            if d.get("is_wicket")
                            else ""
                        )
                    )

            lang_instructions = {
                "ta": "Respond strictly in clear, fluent Tamil (தமிழ்). Keep the tone helpful, natural, and conversational.",
                "hi": "Respond strictly in clear, fluent Hindi (हिन्दी). Keep the tone helpful, natural, and conversational.",
                "en": "Respond strictly in clear, conversational English.",
            }

            system_msg = SystemMessage(
                content=(
                    "You are the Broadcast AI Cricket Expert assisting live match viewers. "
                    "You have access to live game state, recent replay events, scorecards, and player statistics. "
                    "STRICT RULES:\n"
                    "1. Provide clear, accurate, conversational answers grounded strictly in the provided data.\n"
                    "2. Never invent statistics or player numbers.\n"
                    f"3. {lang_instructions.get(lang_key, lang_instructions['en'])}\n"
                    "4. Keep the answer concise (2-3 sentences max)."
                )
            )

            user_content = (
                f"VIEWER QUESTION: '{question}'\n\n"
                f"CURRENT LIVE GAME STATE:\n"
                f"- Batting: {match_state.get('batting_team')} {match_state.get('score')}/{match_state.get('wickets')} in {match_state.get('overs_display')} overs\n"
                f"- Striker: {match_state.get('striker')} ({match_state.get('batters', {}).get(match_state.get('striker'), {}).get('runs', 0)} off {match_state.get('batters', {}).get(match_state.get('striker'), {}).get('balls', 0)})\n"
                f"- Bowler: {match_state.get('bowler')} ({match_state.get('bowlers', {}).get(match_state.get('bowler'), {}).get('wickets', 0)}/{match_state.get('bowlers', {}).get(match_state.get('bowler'), {}).get('runs', 0)})\n"
                f"- Current Run Rate: {match_state.get('crr')}, Required Run Rate: {match_state.get('rrr')}, Target: {match_state.get('target')}\n\n"
                f"RECENT COMPLETED DELIVERIES IN THIS REPLAY SESSION:\n"
                + (
                    "\n".join(recent_events)
                    if recent_events
                    else "No deliveries completed yet."
                )
                + "\n\n"
                f"DATABASE MATCH CONTEXT:\n{json.dumps(match_summary, indent=2)}"
            )

            human_msg = HumanMessage(content=user_content)

            t_qa_start = time.time()
            answer = self._call_groq_messages(
                [system_msg, human_msg],
                model=AgentConfig.PRIMARY_MODEL,
                lang=lang_key,
            )
            elapsed = time.time() - t_qa_start
            MetricsManager.record_qa_latency(elapsed)

            return {
                "answer": answer,
                "language": lang_key,
                "latency_sec": round(elapsed, 3),
                "grounded_state": {
                    "score": f"{match_state.get('score')}/{match_state.get('wickets')}",
                    "overs": match_state.get("overs_display"),
                    "striker": match_state.get("striker"),
                },
            }

        qa_workflow = StateGraph(QAState)
        qa_workflow.add_node("retrieve_context", _retrieve_context)
        qa_workflow.add_node("qa_agent", _qa_agent_node)
        qa_workflow.add_edge(START, "retrieve_context")
        qa_workflow.add_edge("retrieve_context", "qa_agent")
        qa_workflow.add_edge("qa_agent", END)

        return qa_workflow.compile()

    # =====================================================================
    # 5. Public API Facade (100% Backwards Compatible)
    # =====================================================================

    def process_delivery_event(
        self, delivery: dict[str, Any], state: dict[str, Any], lang: str = "en"
    ) -> dict[str, Any]:
        """Invokes the compiled LangGraph StateGraph pipeline for a delivery event."""
        lang_key = self._normalize_lang(lang)
        initial_state: CommentaryState = {
            "delivery": delivery,
            "match_state": state,
            "language": lang_key,
            "reflection_attempts": 0,
            "max_reflection_retries": 2,
            "correction_critique": "",
        }

        # Execute LangGraph Compiled StateGraph
        final_state = self.commentary_graph.invoke(initial_state)

        return {
            "delivery": delivery,
            "language": final_state.get("language", lang_key),
            "lead_commentary": final_state.get("lead_commentary", ""),
            "analyst_commentary": final_state.get("analyst_commentary"),
            "lead_persona": final_state.get("lead_persona", "James Sterling"),
            "analyst_persona": final_state.get(
                "analyst_persona", "Harsha Atherton"
            ),
        }

    def generate_lead_commentary(
        self,
        delivery: dict[str, Any],
        state: dict[str, Any],
        lang: str = "en",
        max_reflection_retries: int = 2,
    ) -> str:
        """Generates dynamic lead commentary via the compiled LangGraph StateGraph."""
        res = self.process_delivery_event(delivery, state, lang=lang)
        return res.get("lead_commentary", "")

    def generate_analyst_commentary(
        self,
        delivery: dict[str, Any],
        state: dict[str, Any],
        lang: str = "en",
        max_reflection_retries: int = 2,
    ) -> str | None:
        """Generates dynamic tactical analyst commentary via the compiled LangGraph StateGraph."""
        res = self.process_delivery_event(delivery, state, lang=lang)
        return res.get("analyst_commentary")

    def answer_viewer_question(
        self,
        question: str,
        state: dict[str, Any],
        lang: str = "en",
        session_deliveries: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Invokes the compiled LangGraph Q&A StateGraph for viewer question."""
        lang_key = self._normalize_lang(lang)
        initial_state: QAState = {
            "question": question,
            "match_state": state,
            "language": lang_key,
            "session_deliveries": session_deliveries or [],
        }

        # Execute LangGraph Q&A Graph
        final_state = self.qa_graph.invoke(initial_state)

        return {
            "question": question,
            "answer": final_state.get("answer", ""),
            "language": final_state.get("language", lang_key),
            "grounded_state": final_state.get("grounded_state", {}),
            "latency_sec": final_state.get("latency_sec", 0.0),
        }


if __name__ == "__main__":
    from engine.replay_engine import ReplayEngine

    engine = ReplayEngine()
    step_result = engine.step()

    graph = CommentaryAgentGraph()
    for l in ["en", "ta", "hi"]:
        res = graph.process_delivery_event(
            step_result["delivery"], step_result["state"], lang=l
        )
        print(f"\n--- Language: {l.upper()} (LangGraph StateGraph) ---")
        print(f"[{res['lead_persona']}]:", res["lead_commentary"])
        if res["analyst_commentary"]:
            print(f"[{res['analyst_persona']}]:", res["analyst_commentary"])
