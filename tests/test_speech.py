"""
Unit tests for Speech Synthesis and Recognition (API-First Cloud Integration).
"""

import os

from speech.asr_engine import ASREngine
from speech.tts_engine import TTSEngine


def test_tts_synthesis():
    tts = TTSEngine()
    audio_path = tts.synthesize_lead("Livingstone with a huge shot down the ground!")
    assert os.path.exists(audio_path)
    assert os.path.getsize(audio_path) > 100


def test_asr_engine():
    asr = ASREngine()
    assert "whisper" in asr.model_size.lower()
    # Verify fallback handling on non-existent audio
    assert asr.transcribe_file("non_existent_path.wav") == ""
