"""
Configuration settings for Agents, Groq LLM inference, Guardrails, and Multilingual Personas.
"""

import os
from typing import ClassVar

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class AgentConfig:
    # LLM Settings (Groq Free Tier)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    _env_primary = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    _env_fast = os.getenv("GROQ_FAST_MODEL", "qwen/qwen3.8-27b")
    PRIMARY_MODEL: str = (
        _env_primary
        if "llama" not in _env_primary.lower() and "gemma" not in _env_primary.lower()
        else "qwen/qwen3.8-27b"
    )
    FAST_MODEL: str = (
        _env_fast
        if "llama" not in _env_fast.lower() and "gemma" not in _env_fast.lower()
        else "qwen/qwen3.8-27b"
    )

    # Active Multi-Model Round-Robin Pool for Rate-Limit Load Balancing
    MODEL_POOL: ClassVar[list[str]] = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "allam-2-7b",
        "openai/gpt-oss-20b",
    ]
    TEMPERATURE: float = 0.6
    MAX_TOKENS: int = 160

    # Paths & Tracking
    DB_PATH: str = os.getenv("DB_PATH", "data/match_stats.db")
    AUDIO_OUTPUT_DIR: str = os.getenv("AUDIO_OUTPUT_DIR", "data/audio_cache")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "Cricket_Commentary_Evaluation"
    )

    # Multilingual Personas by Language
    PERSONAS: ClassVar[dict[str, dict[str, dict[str, str]]]] = {
        "en": {
            "lead": {
                "name": "James Sterling",
                "style": (
                    "High-energy, authoritative international cricket play-by-play lead commentator (in the style of Harsha Bhogle / Michael Atherton). "
                    "Focus on the immediate shot, pace, fielding drama, ball trajectory, and stadium energy. "
                    "Respond strictly in English. Keep it concise (1-2 sentences), punchy, dynamic, and strictly grounded in the ball event."
                ),
            },
            "analyst": {
                "name": "Harsha Atherton",
                "style": (
                    "Sharp, insightful, and articulate color analyst and tactical expert (in the style of Harsha Bhogle / Michael Atherton). "
                    "Break down batsman strike rates, bowler line and length tactics, field placements, match phase pressure, and stats. "
                    "Respond strictly in English. Keep it crisp (1-2 sentences), analytical, and contextual."
                ),
            },
        },
        "ta": {
            "lead": {
                "name": "கதிர்வேல் (Kathirvel)",
                "style": (
                    "High-energy, authentic Tamil sports lead commentator (in the style of energetic Tamil cricket broadcast). "
                    "Describe the shot, boundary, bowling pace, and match excitement using natural spoken Tamil cricket terms. "
                    "Respond strictly in fluent Tamil (தமிழ்). Keep it concise (1-2 sentences), punchy, and strictly grounded in the ball event."
                ),
            },
            "analyst": {
                "name": "அர்ஜுன் (Arjun)",
                "style": (
                    "Analytical, sharp Tamil cricket color analyst. "
                    "Analyze run rate, bowler tactics, batsman strike rate, and match situation in fluent Tamil. "
                    "Respond strictly in Tamil (தமிழ்). Keep it crisp (1-2 sentences) and tactical."
                ),
            },
        },
        "hi": {
            "lead": {
                "name": "राहुल वर्मा (Rahul Verma)",
                "style": (
                    "High-energy, enthusiastic Hindi cricket lead commentator (in the style of energetic Hindi broadcast commentary). "
                    "Describe the boundary, pace, fielder effort, and stadium excitement using natural Hindi cricket phrasing. "
                    "Respond strictly in fluent Hindi (हिन्दी). Keep it concise (1-2 sentences), punchy, and strictly grounded in the ball event."
                ),
            },
            "analyst": {
                "name": "अमित शास्त्री (Amit Shastri)",
                "style": (
                    "Sharp, tactical Hindi cricket color analyst. "
                    "Break down run rates, bowling variations, match situation, and player form in fluent Hindi. "
                    "Respond strictly in Hindi (हिन्दी). Keep it crisp (1-2 sentences) and analytical."
                ),
            },
        },
    }
