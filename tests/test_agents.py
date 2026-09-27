"""
Unit tests for LangGraph Commentary Agents and Guardrails.
"""

from agents.commentary_graph import CommentaryAgentGraph
from agents.guardrails import CommentaryGuardrails
from engine.replay_engine import ReplayEngine


def test_guardrails_validation():
    guardrails = CommentaryGuardrails()

    # 1. Valid test
    valid_deliv = {
        "batter": "LS Livingstone",
        "runs_batter": 1,
        "runs_total": 1,
        "is_wicket": False,
        "extra_type": "none",
    }
    valid_text = "Livingstone drives it gently towards long-on for a single."
    is_valid, _ = guardrails.validate_delivery_commentary(valid_text, valid_deliv)
    assert is_valid is True

    # 2. Hallucinated wicket test
    fake_wicket_text = "Gone! Caught at slips, Livingstone is out!"
    is_valid_wicket, reason = guardrails.validate_delivery_commentary(
        fake_wicket_text, valid_deliv
    )
    assert is_valid_wicket is False
    assert "Hallucination Detected" in reason


def test_agent_commentary_generation():
    engine = ReplayEngine()
    step_res = engine.step()

    graph = CommentaryAgentGraph()
    comm = graph.process_delivery_event(step_res["delivery"], step_res["state"])

    assert "lead_commentary" in comm
    assert len(comm["lead_commentary"]) > 10
    assert comm["lead_persona"] == "James Sterling"


def test_qa_agent():
    engine = ReplayEngine()
    step_res = engine.step()

    graph = CommentaryAgentGraph()
    qa_res = graph.answer_viewer_question(
        "What is the current score?", step_res["state"]
    )

    assert "answer" in qa_res
    assert len(qa_res["answer"]) > 5
