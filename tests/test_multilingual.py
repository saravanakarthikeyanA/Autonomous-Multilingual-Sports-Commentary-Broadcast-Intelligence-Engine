"""
Comprehensive Multilingual Test Suite for English, Tamil, and Hindi Pipelines.
"""

import os

from agents.commentary_graph import CommentaryAgentGraph
from agents.guardrails import CommentaryGuardrails
from engine.payload_builder import build_stream_sync_payload
from engine.replay_engine import ReplayEngine
from speech.tts_engine import TTSEngine


def test_multilingual_guardrails():
    guardrails = CommentaryGuardrails()

    # Test case: Batter scored 1 run, not a wicket, not a six
    delivery = {
        "batter": "LS Livingstone",
        "bowler": "HV Patel",
        "runs_batter": 1,
        "runs_total": 1,
        "is_wicket": False,
        "extra_type": "none",
    }

    # Valid English, Tamil, and Hindi commentary
    valid_en = "Livingstone nudges Patel into the gap for a single."
    valid_ta = "லிவிங்ஸ்டோன் பந்தை தட்டிவிட்டு ஒரு ரன்னை சேர்க்கிறார்."
    valid_hi = "लिविंगस्टोन ने हल्के हाथों से खेलकर एक रन पूरा किया।"

    assert guardrails.validate_delivery_commentary(valid_en, delivery)[0] is True
    assert guardrails.validate_delivery_commentary(valid_ta, delivery)[0] is True
    assert guardrails.validate_delivery_commentary(valid_hi, delivery)[0] is True

    # Hallucinations: claiming wicket when not out
    bad_en = "WICKET! Livingstone is clean bowled!"
    bad_ta = "விக்கெட்! லிவிங்ஸ்டோன் அவுட்!"
    bad_hi = "विकेट! बल्लेबाज आउट होकर पवेलियन लौटे!"

    assert guardrails.validate_delivery_commentary(bad_en, delivery)[0] is False
    assert guardrails.validate_delivery_commentary(bad_ta, delivery)[0] is False
    assert guardrails.validate_delivery_commentary(bad_hi, delivery)[0] is False


def test_multilingual_tts_generation():
    tts = TTSEngine(cache_dir="data/audio_cache")

    for lang in ["en", "ta", "hi"]:
        if lang == "en":
            lead = "Livingstone drives it through the covers for four!"
            analyst = "Superb timing and balance against pace."
        elif lang == "ta":
            lead = "லிவிங்ஸ்டோன் அபாரமான பவுண்டரி அடிக்கிறார்!"
            analyst = "வேகப்பந்து வீச்சை அழகாக கணித்து ஆடிய ஷாட்."
        else:
            lead = "लिविंगस्टोन ने कवर्स की दिशा में शानदार चौका लगाया!"
            analyst = "तेज गति के खिलाफ बेहतरीन संतुलन और टाइमिंग।"

        audio_path, duration = tts.synthesize_dual(lead, analyst, lang=lang)
        assert os.path.exists(audio_path), f"Failed for {lang}"
        assert duration > 0, f"Duration invalid for {lang}"


def test_multilingual_commentary_graph():
    engine = ReplayEngine()
    step_res = engine.step()
    graph = CommentaryAgentGraph()

    for lang in ["en", "ta", "hi"]:
        res = graph.process_delivery_event(
            step_res["delivery"], step_res["state"], lang=lang
        )
        assert res["language"] == lang
        assert res["lead_commentary"] is not None and len(res["lead_commentary"]) > 0
        assert res["lead_persona"] is not None


def test_multilingual_qa_agent():
    engine = ReplayEngine()
    state = engine.get_state_snapshot()
    graph = CommentaryAgentGraph()

    # Test Tamil Q&A
    qa_ta = graph.answer_viewer_question(
        "தற்போதைய ஸ்கோர் மற்றும் பேட்ஸ்மேன் யார்?", state, lang="ta"
    )
    assert qa_ta["language"] == "ta"
    assert len(qa_ta["answer"]) > 0

    # Test Hindi Q&A
    qa_hi = graph.answer_viewer_question(
        "वर्तमान स्कोर और बल्लेबाज कौन है?", state, lang="hi"
    )
    assert qa_hi["language"] == "hi"
    assert len(qa_hi["answer"]) > 0

    # Test English Q&A
    qa_en = graph.answer_viewer_question("What is the current score?", state, lang="en")
    assert qa_en["language"] == "en"
    assert len(qa_en["answer"]) > 0


def test_multilingual_payload_builder():
    engine = ReplayEngine()
    payload = build_stream_sync_payload(engine)
    assert len(payload) > 0
    assert "batter" in payload[0]
    assert "bowler" in payload[0]
    assert "video_time_sec" in payload[0]
    assert "lead_commentary" not in payload[0]
