"""
Autonomous Sports Commentary System - Streamlit Broadcast Platform.
High-end TV Broadcast UI featuring:
1. Real-time Continuous Video Synchronized Replay with instant frame seeking & audio narration.
2. Multilingual Commentary & Voice Support: English (🇬🇧), தமிழ் (🇮🇳), and हिन्दी (🇮🇳).
3. Fixed-height bottom-stacked scrollable AI commentary feed (Zero page displacement).
4. Google Assistant Style Live Microphone Voice Chat with Auto-Spoken Audio in EN / TA / HI.
5. Stock-Market / Financial Terminal Style Momentum & Win-Probability Charts.
6. Prometheus / Grafana Observability integration.
"""

import base64
import hashlib
import os
import time
import warnings

import pandas as pd
import streamlit as st

# Suppress background third-party library warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from agents.commentary_graph import CommentaryAgentGraph
from agents.config import AgentConfig
from engine.payload_builder import build_stream_sync_payload
from engine.replay_engine import ReplayEngine
from monitoring.metrics import MetricsManager
from speech.asr_engine import ASREngine
from speech.tts_engine import TTSEngine
from ui.analytics_charts import (
    calculate_win_probability,
    create_momentum_worm_chart,
    create_win_probability_chart,
)

# Page Setup
st.set_page_config(
    page_title="Multilingual AI Sports Broadcast | Cricket Commentary Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Start Prometheus Metrics Exporter
if "metrics_started" not in st.session_state:
    MetricsManager.start_exporter(8000)
    st.session_state["metrics_started"] = True

# Custom Broadcast Theme CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(145deg, #070B19 0%, #0D1630 50%, #091024 100%);
        color: #F0F4F8;
    }
    
    /* Live Match Header Banner */
    .match-banner {
        background: linear-gradient(90deg, rgba(14,28,64,0.95) 0%, rgba(20,40,90,0.95) 100%);
        border: 1px solid rgba(0, 240, 255, 0.35);
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }
    
    .live-badge {
        background-color: #FF0055;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.78rem;
        letter-spacing: 1.5px;
        animation: pulse 1.5s infinite;
        display: inline-block;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.98); opacity: 0.85; }
        50% { transform: scale(1.03); opacity: 1; box-shadow: 0 0 16px #FF0055; }
        100% { transform: scale(0.98); opacity: 0.85; }
    }
    
    .score-box {
        font-size: 2.1rem;
        font-weight: 900;
        color: #00F0FF;
        line-height: 1.1;
    }
    
    .score-sub {
        color: #A0AEC0;
        font-size: 0.9rem;
        margin-top: 4px;
    }

    .lang-pill {
        background: rgba(0, 240, 255, 0.15);
        border: 1px solid #00F0FF;
        color: #00F0FF;
        border-radius: 16px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
    }
</style>
""",
    unsafe_allow_html=True,
)

# State Initializations
if "selected_language" not in st.session_state:
    st.session_state["selected_language"] = "en"

if "replay_engine" not in st.session_state:
    st.session_state["replay_engine"] = ReplayEngine()

if "agent_graph" not in st.session_state:
    st.session_state["agent_graph"] = CommentaryAgentGraph()

if "tts_engine" not in st.session_state:
    st.session_state["tts_engine"] = TTSEngine()

if "asr_engine" not in st.session_state:
    st.session_state["asr_engine"] = ASREngine()

if "qa_chat_history" not in st.session_state:
    st.session_state["qa_chat_history"] = []

if "prob_history" not in st.session_state:
    st.session_state["prob_history"] = [
        {"over_ball": 18.0, "england": 65.0, "india": 35.0}
    ]

if "completed_deliveries" not in st.session_state:
    st.session_state["completed_deliveries"] = []

if "sync_payload" not in st.session_state:
    st.session_state["sync_payload"] = build_stream_sync_payload(
        st.session_state["replay_engine"]
    )

if "stream_idx" not in st.session_state:
    st.session_state["stream_idx"] = 0

if "is_live_playing" not in st.session_state:
    st.session_state["is_live_playing"] = False

if "delivery_phase" not in st.session_state:
    st.session_state["delivery_phase"] = "IDLE"

if "display_state" not in st.session_state:
    st.session_state["display_state"] = st.session_state[
        "replay_engine"
    ].get_state_snapshot()

if "commentary_cards" not in st.session_state:
    st.session_state["commentary_cards"] = []

if "last_processed_audio_hash" not in st.session_state:
    st.session_state["last_processed_audio_hash"] = None

if "active_audio_path" not in st.session_state:
    st.session_state["active_audio_path"] = None

if "current_video_time" not in st.session_state:
    st.session_state["current_video_time"] = 24

if "pending_delivery_package" not in st.session_state:
    st.session_state["pending_delivery_package"] = None

if "last_audio_dur" not in st.session_state:
    st.session_state["last_audio_dur"] = 5.0

engine: ReplayEngine = st.session_state["replay_engine"]
agent_graph: CommentaryAgentGraph = st.session_state["agent_graph"]
tts: TTSEngine = st.session_state["tts_engine"]
asr: ASREngine = st.session_state["asr_engine"]
active_lang = st.session_state.get("selected_language", "en")


def start_delivery_action():
    """
    Phase 1 (Generation Time / Ball Release):
    - Starts video at ball release timestamp.
    - Suppresses spoiler commentary & score updates.
    - Concurrently in background: runs LangGraph multi-agent generation (Lead + Analyst)
      and synthesizes dual-commentator neural audio so everything is 100% ready before reveal.
    """
    payload = st.session_state["sync_payload"]
    idx = st.session_state["stream_idx"]
    if idx < len(payload):
        d_payload = payload[idx]
        v_sec = d_payload.get("video_time_sec")
        if v_sec is not None:
            st.session_state["current_video_time"] = int(v_sec)
        st.session_state["delivery_phase"] = "ACTION"
        st.session_state["active_audio_path"] = None

        # --- Phase 1 Background Generation ---
        cur_lang = st.session_state.get("selected_language", "en")
        personas = AgentConfig.PERSONAS.get(cur_lang, AgentConfig.PERSONAS["en"])
        current_snap = engine.get_state_snapshot()

        # 1. Dynamic commentary generation with LangGraph multi-agent & guardrails
        lead_text = agent_graph.generate_lead_commentary(
            d_payload, current_snap, lang=cur_lang
        )
        analyst_text = (
            agent_graph.generate_analyst_commentary(
                d_payload, current_snap, lang=cur_lang
            )
            or ""
        )

        # 2. Synthesize dual-voice neural audio on-the-fly during action window
        audio_path, audio_dur = tts.synthesize_dual(
            lead_text, analyst_text, lang=cur_lang
        )

        # 3. Store in pending delivery package
        st.session_state["pending_delivery_package"] = {
            "over": d_payload["over"],
            "ball": d_payload["ball"],
            "runs": d_payload["runs_total"],
            "is_wicket": d_payload["is_wicket"],
            "language": cur_lang,
            "lead_persona": personas["lead"]["name"],
            "analyst_persona": personas["analyst"]["name"],
            "lead": lead_text,
            "analyst": analyst_text,
            "audio_path": audio_path,
            "audio_dur": audio_dur,
        }


def complete_delivery_outcome():
    """
    Phase 2 (Reveal Time / Outcome at 8.0s):
    - Steps match state and reveals scoreboard update.
    - Reveals commentary card in feed.
    - Triggers instant, zero-latency spoken commentary voice-over (already synthesized in Phase 1).
    """
    payload = st.session_state["sync_payload"]
    idx = st.session_state["stream_idx"]
    if idx < len(payload):
        d_payload = payload[idx]
        engine.step()

        # 1. Update Display State to post-delivery outcome
        st.session_state["display_state"] = engine.get_state_snapshot()
        st.session_state["delivery_phase"] = "OUTCOME"

        snap = st.session_state["display_state"]
        MetricsManager.record_delivery(
            match_id="1276906",
            inning=snap.get("inning", 1),
            runs=snap.get("score", 190),
            wickets=snap.get("wickets", 5),
            team=snap.get("batting_team", "ENG"),
        )

        # 2. Retrieve the pre-generated package from Phase 1
        pkg = st.session_state.get("pending_delivery_package")
        if not pkg:
            cur_lang = st.session_state.get("selected_language", "en")
            personas = AgentConfig.PERSONAS.get(cur_lang, AgentConfig.PERSONAS["en"])
            lead_text = agent_graph.generate_lead_commentary(
                d_payload, snap, lang=cur_lang
            )
            analyst_text = (
                agent_graph.generate_analyst_commentary(d_payload, snap, lang=cur_lang)
                or ""
            )
            audio_path, audio_dur = tts.synthesize_dual(
                lead_text, analyst_text, lang=cur_lang
            )
            pkg = {
                "over": d_payload["over"],
                "ball": d_payload["ball"],
                "runs": d_payload["runs_total"],
                "is_wicket": d_payload["is_wicket"],
                "language": cur_lang,
                "lead_persona": personas["lead"]["name"],
                "analyst_persona": personas["analyst"]["name"],
                "lead": lead_text,
                "analyst": analyst_text,
                "audio_path": audio_path,
                "audio_dur": audio_dur,
            }

        # 3. Append Commentary Card
        st.session_state["commentary_cards"].append(
            {
                "over": pkg["over"],
                "ball": pkg["ball"],
                "runs": pkg["runs"],
                "is_wicket": pkg["is_wicket"],
                "language": pkg["language"],
                "lead_persona": pkg["lead_persona"],
                "analyst_persona": pkg["analyst_persona"],
                "lead": pkg["lead"],
                "analyst": pkg["analyst"],
                "audio_path": pkg["audio_path"],
            }
        )
        st.session_state["completed_deliveries"].append(pkg)

        # 4. Trigger Spoken Voice-Over Automatically (Zero Latency)
        st.session_state["active_audio_path"] = pkg["audio_path"]
        st.session_state["last_audio_dur"] = pkg["audio_dur"]
        st.session_state["pending_delivery_package"] = None

        # 5. Update Win Probability History
        c_state = st.session_state["display_state"]
        probs = calculate_win_probability(c_state)
        st.session_state["prob_history"].append(
            {
                "over_ball": float(c_state["overs_display"]),
                "england": probs["batting_team_prob"]
                if c_state["inning"] == 1
                else probs["bowling_team_prob"],
                "india": probs["bowling_team_prob"]
                if c_state["inning"] == 1
                else probs["batting_team_prob"],
            }
        )

        st.session_state["stream_idx"] += 1


def advance_single_delivery():
    """Advances one delivery with two-phase control."""
    phase = st.session_state.get("delivery_phase", "IDLE")
    if phase == "ACTION":
        complete_delivery_outcome()
    else:
        start_delivery_action()


def reset_stream_state():
    st.session_state["replay_engine"].reset_to_live_start()
    st.session_state["stream_idx"] = 0
    st.session_state["delivery_phase"] = "IDLE"
    st.session_state["display_state"] = st.session_state[
        "replay_engine"
    ].get_state_snapshot()
    st.session_state["is_live_playing"] = False
    st.session_state["commentary_cards"] = []
    st.session_state["completed_deliveries"] = []
    st.session_state["active_audio_path"] = None
    st.session_state["pending_delivery_package"] = None
    st.session_state["current_video_time"] = 24
    st.session_state["prob_history"] = [
        {"over_ball": 18.0, "england": 65.0, "india": 35.0}
    ]


# Sidebar / Header Controls & Multilingual Language Selector
with st.sidebar:
    st.markdown("### 🌐 Broadcast Settings")

    lang_options = {
        "🇬🇧 English": "en",
        "🇮🇳 தமிழ் (Tamil)": "ta",
        "🇮🇳 हिन्दी (Hindi)": "hi",
    }

    current_lang_label = [k for k, v in lang_options.items() if v == active_lang][0]
    selected_lang_label = st.selectbox(
        "🎙️ Commentary Language",
        options=list(lang_options.keys()),
        index=list(lang_options.keys()).index(current_lang_label),
    )
    new_lang = lang_options[selected_lang_label]

    if new_lang != active_lang:
        st.session_state["selected_language"] = new_lang
        MetricsManager.set_active_language(new_lang)
        # Rebuild sync payload for selected language
        st.session_state["sync_payload"] = build_stream_sync_payload(
            st.session_state["replay_engine"],
            st.session_state["tts_engine"],
            lang=new_lang,
        )
        st.rerun()

    active_personas = AgentConfig.PERSONAS.get(new_lang, AgentConfig.PERSONAS["en"])
    st.markdown(f"""
    **🎙️ Active Commentators:**
    - **Lead:** `{active_personas["lead"]["name"]}`
    - **Analyst:** `{active_personas["analyst"]["name"]}`
    """)
    st.divider()
    st.caption(
        "⚡ Powered by Faster-Whisper ASR, Multilingual Neural TTS & LangGraph Multi-Agent Orchestration."
    )


current_state = st.session_state.get("display_state", engine.get_state_snapshot())
phase = st.session_state.get("delivery_phase", "IDLE")
payload = st.session_state["sync_payload"]
cur_idx = st.session_state["stream_idx"]
active_d = payload[cur_idx] if cur_idx < len(payload) else None

# Header Scoreboard Banner
target_display = (
    f"Target: {current_state['target']} | RRR: {current_state['rrr']}"
    if current_state["target"]
    else f"CRR: {current_state['crr']}"
)
badge_html = (
    '<span class="live-badge">● LIVE BROADCAST</span>'
    if st.session_state["is_live_playing"]
    else '<span class="live-badge" style="background:#718096; animation:none;">⏸ PAUSED</span>'
)
lang_tag = {"en": "🇬🇧 English Feed", "ta": "🇮🇳 தமிழ் வர்ணனை", "hi": "🇮🇳 हिन्दी कमेंट्री"}.get(
    active_lang, "🇬🇧 English Feed"
)

st.markdown(
    f"""
<div class="match-banner">
    <div>
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
            {badge_html}
            <span class="lang-pill">{lang_tag}</span>
            <span style="font-weight:700; color:#E2E8F0; font-size:1.1rem;">3rd T20I: England vs India</span>
            <span style="color:#718096; font-size:0.85rem;">Trent Bridge, Nottingham</span>
        </div>
        <div class="score-sub">Batting: <strong style="color:white;">{current_state["batting_team"]}</strong> | Striker: <strong style="color:#00F0FF;">{current_state["striker"]}</strong> | Bowler: <strong style="color:#FFB800;">{current_state["bowler"]}</strong></div>
    </div>
    <div style="text-align:right;">
        <div class="score-box">{current_state["score"]}/{current_state["wickets"]} <span style="font-size:1.1rem; color:#A0AEC0;">({current_state["overs_display"]} ov)</span></div>
        <div style="color:#00E5FF; font-weight:600; font-size:0.88rem;">{target_display}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# Main Layout (2 Columns: Left = Continuous Video Broadcast Stage, Right = AI Assistant & Analytics)
col_left, col_right = st.columns([7, 5], gap="medium")

# --- LEFT COLUMN: Broadcast Stage with Real-Time Synchronized Player ---
with col_left:
    st.markdown("### 📺 Live Broadcast Stage")

    # 1. Native High-Performance Video Player
    video_start = int(st.session_state.get("current_video_time", 24))
    st.video(
        "data/1276906.mp4",
        start_time=video_start,
        autoplay=st.session_state.get("is_live_playing", False),
    )

    # Helper for broadcast TV event formatting
    def format_last_event(event_dict, start_sec, cur_phase, cur_d, cur_lang):
        if cur_phase == "ACTION" and cur_d:
            if cur_lang == "ta":
                return f"🟡 பந்து வீசப்படுகிறது: ஓவர் {cur_d['over']}.{cur_d['ball']} ({cur_d['bowler']} vs {cur_d['batter']})"
            elif cur_lang == "hi":
                return f"🟡 गेंद डाली जा रही है: ओवर {cur_d['over']}.{cur_d['ball']} ({cur_d['bowler']} vs {cur_d['batter']})"
            return f"🟡 BALL IN PLAY: Over {cur_d['over']}.{cur_d['ball']} ({cur_d['bowler']} to {cur_d['batter']})"
        if not event_dict or not isinstance(event_dict, dict):
            return f"Ready @ {start_sec}.0s"
        ov = event_dict.get("over", 18)
        ball = event_dict.get("ball", 0)
        runs = event_dict.get("runs_total", 0)
        batter = event_dict.get("batter", "")
        bowler = event_dict.get("bowler", "")
        is_w = event_dict.get("is_wicket", False)
        p_out = event_dict.get("player_out") or batter
        w_kind = event_dict.get("wicket_kind", "out")

        if is_w:
            return f"🔴 OUT! {p_out} ({w_kind}) b {bowler}"
        elif runs == 6:
            return f"💥 SIX! {batter} (6 runs)"
        elif runs == 4:
            return f"⚡ FOUR! {batter} (4 runs)"
        elif event_dict.get("extra_type") == "wides":
            return f"⚠️ Wide ({runs} run{'s' if runs > 1 else ''})"
        elif event_dict.get("extra_type") == "noballs":
            return f"⚠️ No Ball ({runs} run{'s' if runs > 1 else ''})"
        else:
            return f"Over {ov}.{ball}: {runs} run{'s' if runs != 1 else ''} ({batter})"

    # 2. Real-Time On-Screen TV Score Ribbon
    last_event_text = format_last_event(
        current_state.get("last_event"), video_start, phase, active_d, active_lang
    )
    st.markdown(
        f"""
    <div style="background: rgba(7, 11, 25, 0.95); border-left: 4px solid #00F0FF; border-radius: 8px; padding: 10px 14px; margin-top: -6px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.6);">
        <div>
            <div style="font-weight: 800; color: #00F0FF; font-size: 1.05rem;">{current_state["batting_team"]} {current_state["score"]}/{current_state["wickets"]} ({current_state["overs_display"]} ov)</div>
            <div style="font-size: 0.8rem; color: #E2E8F0;">🏏 {current_state["striker"]} | 🎯 {current_state["bowler"]}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.82rem; color: #FFD700; font-weight: 700;">{last_event_text}</div>
            <div style="font-size: 0.72rem; color: #718096;">● Synced @ {video_start}.0s</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 3. Broadcast Control Deck
    btn_c1, btn_c2, btn_c3 = st.columns(3)
    with btn_c1:
        if st.session_state.get("is_live_playing", False):
            if st.button(
                "⏸ Pause Live Stream", use_container_width=True, type="primary"
            ):
                st.session_state["is_live_playing"] = False
                st.rerun()
        else:
            if st.button(
                "▶ Play Live Stream", use_container_width=True, type="primary"
            ):
                st.session_state["is_live_playing"] = True
                if st.session_state["stream_idx"] == 0 and st.session_state.get(
                    "delivery_phase"
                ) in ["IDLE", "OUTCOME"]:
                    start_delivery_action()
                st.rerun()

    with btn_c2:
        if st.button("⏭ Next Delivery", use_container_width=True):
            advance_single_delivery()
            st.rerun()

    with btn_c3:
        if st.button("🔄 Reset (Over 18.0)", use_container_width=True):
            reset_stream_state()
            st.rerun()

    # 4. Live Commentary Stream Container with Match Expert Chat Style Auto-Playback
    active_personas = AgentConfig.PERSONAS.get(active_lang, AgentConfig.PERSONAS["en"])
    lead_name = active_personas["lead"]["name"]
    analyst_name = active_personas["analyst"]["name"]

    st.markdown(
        f"<div style='font-size:0.85rem; font-weight:800; color:#00F0FF; letter-spacing:0.8px; margin-top:10px; margin-bottom:6px;'>🎙️ REAL-TIME AI COMMENTARY ({lead_name.upper()} & {analyst_name.upper()})</div>",
        unsafe_allow_html=True,
    )
    with st.container(height=300):
        commentary_cards = st.session_state.get("commentary_cards", [])
        if not commentary_cards:
            st.info(
                f"Broadcast ready at Over 18.0 ({video_start}.0s). Click **Play Live Stream** or **Next Delivery** to begin {lang_tag}."
            )

        total_cards = len(commentary_cards)
        for card_idx, card in enumerate(reversed(commentary_cards)):
            original_idx = total_cards - 1 - card_idx
            is_latest_delivery = (original_idx == total_cards - 1) and (
                st.session_state.get("delivery_phase") == "OUTCOME"
            )

            card_lead_persona = card.get("lead_persona", lead_name)
            card_analyst_persona = card.get("analyst_persona", analyst_name)
            st.markdown(
                f"""
            <div style="background: rgba(0, 229, 255, 0.08); border-left: 4px solid #00F0FF; border-radius: 6px; padding: 8px 10px; margin-bottom: 6px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                    <span style="font-size:0.75rem; font-weight:800; color:#00F0FF;">🎙️ {card_lead_persona}</span>
                    <span style="font-size:0.7rem; color:#A0AEC0;">Over {card["over"]}.{card["ball"]} • {card["runs"]} runs</span>
                </div>
                <div style="font-size:0.84rem; color:#FFFFFF;">{card["lead"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if card.get("analyst"):
                st.markdown(
                    f"""
                <div style="background: rgba(255, 184, 0, 0.08); border-left: 4px solid #FFB800; border-radius: 6px; padding: 8px 10px; margin-bottom: 6px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                        <span style="font-size:0.75rem; font-weight:800; color:#FFB800;">📊 {card_analyst_persona}</span>
                        <span style="font-size:0.7rem; color:#A0AEC0;">Tactical Insight</span>
                    </div>
                    <div style="font-size:0.84rem; color:#FFFFFF;">{card["analyst"]}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Exact Match Expert Chat style audio player with native autoplay
            if card.get("audio_path") and os.path.exists(card["audio_path"]):
                st.audio(
                    card["audio_path"], format="audio/wav", autoplay=is_latest_delivery
                )

    # Active Delivery Automatic Voice-Over Trigger
    active_audio = st.session_state.get("active_audio_path")
    if (
        st.session_state.get("delivery_phase") == "OUTCOME"
        and active_audio
        and os.path.exists(active_audio)
    ):
        try:
            with open(active_audio, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode("utf-8")
            stream_idx = st.session_state.get("stream_idx", 0)
            st.components.v1.html(
                f"""
                <audio id="live_delivery_audio_{stream_idx}" autoplay preload="auto" playsinline style="display:none;">
                    <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                </audio>
                <script>
                    (function() {{
                        const player = document.getElementById("live_delivery_audio_{stream_idx}");
                        if (player) {{
                            player.volume = 1.0;
                            player.play().catch(function(err) {{
                                console.log("[BroadcastVoice] Autoplay notice:", err);
                            }});
                        }}
                    }})();
                </script>
                """,
                height=0,
                width=0,
            )
        except Exception as e:
            print(f"[BroadcastVoice] Error: {e}")


# --- RIGHT COLUMN: Google-Assistant Voice Chat & Analytics ---
with col_right:
    tab_qa, tab_analytics, tab_scorecard = st.tabs(
        ["🗣️ Match Expert Chat", "📈 Financial Analytics", "📊 Scorecard"]
    )

    # TAB 1: Live Voice & Text Match Chat in Active Language
    with tab_qa:
        st.markdown(f"#### 🗣️ Multilingual Live Voice AI Expert ({lang_tag})")
        st.caption(
            "Speak into your microphone or type in English, தமிழ், or हिन्दी for instant spoken answers!"
        )

        # Native Live Microphone Audio Input
        live_audio = None
        try:
            live_audio = st.audio_input("🎙️ Tap to Speak (Auto-Transcribe EN / TA / HI)")
        except AttributeError:
            live_audio = st.file_uploader(
                "🎙️ Upload Voice Question (.wav)", type=["wav", "mp3"]
            )

        if live_audio is not None:
            audio_bytes = (
                live_audio.read()
                if hasattr(live_audio, "read")
                else live_audio.getvalue()
            )
            if audio_bytes:
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                if audio_hash != st.session_state.get("last_processed_audio_hash"):
                    st.session_state["last_processed_audio_hash"] = audio_hash
                    with st.spinner(
                        "🎧 Transcribing live speech with multilingual Whisper..."
                    ):
                        t_asr = time.time()
                        transcribed_text = asr.transcribe_bytes(
                            audio_bytes, language=active_lang
                        )
                        MetricsManager.record_asr_latency(time.time() - t_asr)

                    if transcribed_text:
                        st.success(f'🗣️ **You Said:** *"{transcribed_text}"*')
                        with st.spinner(
                            f"🤖 Reasoning & synthesizing answer in {active_lang.upper()}..."
                        ):
                            qa_res = agent_graph.answer_viewer_question(
                                transcribed_text,
                                current_state,
                                lang=active_lang,
                                session_deliveries=st.session_state.get(
                                    "completed_deliveries", []
                                ),
                            )
                            ans_audio = tts.synthesize_analyst(
                                qa_res["answer"], lang=active_lang
                            )
                            MetricsManager.record_qa_question(input_type="voice")
                            st.session_state["just_asked"] = True

                            st.session_state["qa_chat_history"].append(
                                {
                                    "question": transcribed_text,
                                    "answer": qa_res["answer"],
                                    "audio": ans_audio,
                                }
                            )

        # Text Query Option
        with st.form(key="text_qa_form_v5", clear_on_submit=True):
            placeholder_text = {
                "en": "e.g. What is the required run rate and who is batting?",
                "ta": "எ.கா. தற்போதைய ரன் ரேட் என்ன? யார் பேட்டிங் செய்கிறார்கள்?",
                "hi": "उदा. वर्तमान आवश्यक रन रेट क्या है और कौन बल्लेबाजी कर रहा है?",
            }.get(active_lang, "e.g. What is the required run rate?")

            user_question = st.text_input(
                "💬 Or type your query (EN / தமிழ் / हिन्दी):",
                placeholder=placeholder_text,
            )
            submit_btn = st.form_submit_button("Ask Assistant")

            if submit_btn and user_question.strip():
                with st.spinner(
                    f"🤖 Querying match database in {active_lang.upper()}..."
                ):
                    qa_res = agent_graph.answer_viewer_question(
                        user_question,
                        current_state,
                        lang=active_lang,
                        session_deliveries=st.session_state.get(
                            "completed_deliveries", []
                        ),
                    )
                    ans_audio = tts.synthesize_analyst(
                        qa_res["answer"], lang=active_lang
                    )
                    MetricsManager.record_qa_question(input_type="text")
                    st.session_state["just_asked"] = True
                    st.session_state["qa_chat_history"].append(
                        {
                            "question": user_question,
                            "answer": qa_res["answer"],
                            "audio": ans_audio,
                        }
                    )

        # Assistant Chat Stream with Fixed Height Container
        with st.container(height=360):
            if not st.session_state["qa_chat_history"]:
                st.info(
                    "Ask any cricket question above via voice or text. The assistant will respond and speak aloud in your selected language."
                )
            total_chats = len(st.session_state["qa_chat_history"])
            for c_idx, chat in enumerate(st.session_state["qa_chat_history"]):
                st.markdown(f"**👤 Fan:** {chat['question']}")
                st.markdown(f"**🤖 Assistant:** {chat['answer']}")
                if chat.get("audio") and os.path.exists(chat["audio"]):
                    is_latest_response = (
                        c_idx == total_chats - 1
                    ) and st.session_state.get("just_asked", False)
                    st.audio(
                        chat["audio"], format="audio/wav", autoplay=is_latest_response
                    )
                st.divider()

    # TAB 2: Stock-Market Style Financial Terminal Analytics
    with tab_analytics:
        st.markdown("#### 📈 Live Match Momentum & Market Velocity")

        # 1. Live Momentum Worm Chart
        sample_history = []
        for p_item in st.session_state["sync_payload"]:
            sample_history.append(
                {
                    "deliv": {
                        "over": p_item["over"],
                        "ball": p_item["ball"],
                        "runs_total": p_item["runs_total"],
                        "runs_batter": p_item["runs_total"],
                        "is_wicket": p_item["is_wicket"],
                    },
                    "state": {"score": int(p_item["score"].split("/")[0])},
                }
            )
        worm_chart = create_momentum_worm_chart(sample_history)
        st.altair_chart(worm_chart, width="stretch")

        # 2. Live Win Probability Ticker Chart
        prob_chart = create_win_probability_chart(st.session_state["prob_history"])
        st.altair_chart(prob_chart, width="stretch")

        # 3. Live Velocity Meters
        probs = calculate_win_probability(current_state)
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.metric(
                "🏆 England Win %",
                f"{probs['batting_team_prob'] if current_state['inning'] == 1 else probs['bowling_team_prob']}%",
            )
        with v_col2:
            st.metric(
                "🇮🇳 India Win %",
                f"{probs['bowling_team_prob'] if current_state['inning'] == 1 else probs['batting_team_prob']}%",
            )

    # TAB 3: Full Batting & Bowling Scorecard
    with tab_scorecard:
        st.markdown("#### 🏏 Batting Scorecard")
        batters_list = []
        for name, b in current_state.get("batters", {}).items():
            sr = round(b["runs"] / b["balls"] * 100.0, 1) if b["balls"] > 0 else 0.0
            batters_list.append(
                {
                    "Batter": name,
                    "Runs": b["runs"],
                    "Balls": b["balls"],
                    "4s": b["fours"],
                    "6s": b["sixes"],
                    "SR": sr,
                    "Dismissal": b.get("dismissal", "not out"),
                }
            )
        if batters_list:
            st.dataframe(pd.DataFrame(batters_list), hide_index=True, width="stretch")

        st.markdown("#### 🎯 Bowling Figures")
        bowlers_list = []
        for name, bw in current_state.get("bowlers", {}).items():
            ov = (bw["legal_balls"] // 6) + (bw["legal_balls"] % 6) / 10.0
            econ = (
                round(bw["runs"] / (bw["legal_balls"] / 6.0), 2)
                if bw["legal_balls"] > 0
                else 0.0
            )
            bowlers_list.append(
                {
                    "Bowler": name,
                    "Overs": ov,
                    "Maidens": bw["maidens"],
                    "Runs": bw["runs"],
                    "Wickets": bw["wickets"],
                    "Economy": econ,
                }
            )
        if bowlers_list:
            st.dataframe(pd.DataFrame(bowlers_list), hide_index=True, width="stretch")

# Auto-Progression Event Loop with Two-Phase Broadcast Timeline Synchronization
if st.session_state.get("is_live_playing", False):
    payload = st.session_state["sync_payload"]
    cur_idx = st.session_state["stream_idx"]
    cur_phase = st.session_state.get("delivery_phase", "IDLE")

    if cur_phase == "ACTION":
        # Phase 1: Wait 8.0s live action
        time.sleep(8.0)
        complete_delivery_outcome()
        st.rerun()
    elif cur_phase in ["OUTCOME", "IDLE"]:
        if cur_idx < len(payload):
            cur_d = payload[max(0, cur_idx - 1)] if cur_idx > 0 else payload[0]
            next_d = payload[cur_idx]

            audio_dur = float(st.session_state.get("last_audio_dur") or 6.0)
            wait_time = max(audio_dur + 1.0, 5.0)

            time.sleep(wait_time)
            start_delivery_action()
            st.rerun()
        else:
            st.session_state["is_live_playing"] = False
            st.balloons()
            st.success("🏁 3rd T20I Match Replay Completed!")
