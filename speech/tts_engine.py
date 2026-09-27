"""
Multilingual Dual-Persona Text-to-Speech Engine supporting English, Tamil, and Hindi.
Provides distinct broadcast voices for Lead Commentator and Color Analyst
with disk-based audio caching and thread-safe async execution for Streamlit compatibility.
Optimized for AWS Free Tier with API-First Cloud Synthesis (Edge-TTS, Hugging Face Serverless, gTTS).
"""

import asyncio
import concurrent.futures
import hashlib
import os
import time
import warnings

import numpy as np
import requests
import soundfile as sf

from monitoring.metrics import MetricsManager

# Suppress internal warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    from gtts import gTTS

    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


def _run_async_safe(coro, timeout: float = 20.0):
    """
    Safely executes an async coroutine inside a fresh, dedicated worker thread
    with its own event loop. This avoids 'RuntimeError: This event loop is already running'
    when invoked from Streamlit or Tornado async threads.
    """

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker)
        return future.result(timeout=timeout)


class TTSEngine:
    # Voice profiles per language for Lead Commentator and Color Analyst
    VOICE_REGISTRY: dict[str, dict[str, str]] = {
        "en": {
            "lead": "en-IN-PrabhatNeural",  # High-energy Indian English broadcast voice
            "analyst": "en-GB-RyanNeural",  # Distinct analytical British/Intl English voice
            "hf_lead": "am_adam",
            "hf_analyst": "bm_george",
        },
        "ta": {
            "lead": "ta-IN-ValluvarNeural",  # Authoritative, high-energy Tamil broadcast voice
            "analyst": "ta-IN-PallaviNeural",  # Articulate, smooth Tamil commentary voice
        },
        "hi": {
            "lead": "hi-IN-MadhurNeural",  # Energetic Hindi sports commentary voice
            "analyst": "hi-IN-SwaraNeural",  # Clear, analytical Hindi voice
        },
    }

    # Hugging Face Kokoro-82M Serverless Inference Endpoint
    HF_KOKORO_API_URL = "https://api-inference.huggingface.co/models/hexgrad/Kokoro-82M"

    def __init__(self, cache_dir: str = "data/audio_cache", default_lang: str = "en"):
        self.cache_dir = cache_dir
        self.default_lang = default_lang
        self.hf_token = os.environ.get("HF_TOKEN", "").strip()
        os.makedirs(self.cache_dir, exist_ok=True)
        print("[TTSEngine] Initialized in API-First mode (Zero-RAM Cloud TTS).")

    def _normalize_lang(self, lang: str | None) -> str:
        if not lang:
            return self.default_lang
        l = lang.lower().strip()
        if "ta" in l or "tamil" in l or "தமிழ்" in l:
            return "ta"
        if "hi" in l or "hindi" in l or "हिन्दी" in l or "हिंदी" in l:
            return "hi"
        return "en"

    def _get_cache_path(self, text: str, voice: str, lang: str) -> str:
        """Computes deterministic hash path for caching with language prefix."""
        hash_str = hashlib.md5(f"{lang}:{voice}:{text}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{lang}_{voice}_{hash_str}.wav")

    def _sanitize_tts_text(self, text: str) -> str:
        """Removes markdown symbols, emojis, and unprintable characters that could disrupt TTS encoders."""
        import re

        # Remove markdown emphasis, headings, and quotes
        cleaned = re.sub(r"[\*\#\_`~>]", " ", text)
        # Normalize multiple spaces and whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _synthesize_edge(
        self, text: str, voice: str, output_path: str, rate: str = "+0%"
    ) -> bool:
        """Synthesizes speech using edge-tts safely in an isolated event loop with backup voice failover."""
        if not EDGE_TTS_AVAILABLE:
            return False

        clean_text = self._sanitize_tts_text(text)
        if not clean_text:
            return False

        # Define candidate voices (primary + backup)
        candidate_voices = [voice]
        if "Ryan" in voice:
            candidate_voices.append("en-GB-SoniaNeural")
            candidate_voices.append("en-GB-ThomasNeural")
        elif "Prabhat" in voice:
            candidate_voices.append("en-IN-NeerjaNeural")

        for v in candidate_voices:
            try:
                tmp_mp3 = output_path.replace(
                    ".wav", f"_{os.getpid()}_{hash(v) % 10000}_edge.mp3"
                )

                async def _do_edge(selected_v=v):
                    communicate = edge_tts.Communicate(
                        clean_text, selected_v, rate=rate
                    )
                    await communicate.save(tmp_mp3)

                _run_async_safe(_do_edge(v), timeout=15.0)

                if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 100:
                    data, sr = sf.read(tmp_mp3)
                    sf.write(output_path, data, sr)
                    try:
                        os.remove(tmp_mp3)
                    except OSError:
                        pass
                    return True
            except Exception as e:
                print(
                    f"[TTSEngine] Edge-TTS notice for '{v}': {e}. Trying next candidate..."
                )
                continue
        return False

    def _synthesize_hf_kokoro(self, text: str, voice: str, output_path: str) -> bool:
        """Synthesizes speech using Hugging Face Serverless Inference API for Kokoro-82M."""
        headers = {}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        clean_text = self._sanitize_tts_text(text)
        if not clean_text:
            return False

        try:
            payload = {"inputs": clean_text, "parameters": {"voice": voice}}
            response = requests.post(
                self.HF_KOKORO_API_URL, headers=headers, json=payload, timeout=10
            )
            if response.status_code == 200 and len(response.content) > 500:
                tmp_audio = output_path.replace(".wav", f"_{os.getpid()}_hf.wav")
                with open(tmp_audio, "wb") as f:
                    f.write(response.content)
                if os.path.exists(tmp_audio) and os.path.getsize(tmp_audio) > 100:
                    data, sr = sf.read(tmp_audio)
                    sf.write(output_path, data, sr)
                    try:
                        os.remove(tmp_audio)
                    except OSError:
                        pass
                    return True
            elif response.status_code != 200:
                print(
                    f"[TTSEngine] Hugging Face Inference notice: HTTP {response.status_code}"
                )
        except Exception as e:
            print(f"[TTSEngine] Hugging Face Kokoro synthesis notice: {e}")
        return False

    def _synthesize_gtts(self, text: str, lang: str, output_path: str) -> bool:
        """Synthesizes speech using gTTS as resilient multilingual fallback."""
        if not GTTS_AVAILABLE:
            return False
        try:
            tmp_mp3 = output_path.replace(".wav", f"_{os.getpid()}_gtts.mp3")
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(tmp_mp3)
            if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 100:
                data, sr = sf.read(tmp_mp3)
                sf.write(output_path, data, sr)
                try:
                    os.remove(tmp_mp3)
                except OSError:
                    pass
                return True
        except Exception as e:
            print(f"[TTSEngine] gTTS notice for '{lang}': {e}")
        return False

    @staticmethod
    def is_valid_speech_audio(file_path: str | None) -> bool:
        """Validates that audio file exists, has real content, and is NOT a synthetic beep tone."""
        if not file_path or not os.path.exists(file_path):
            return False
        try:
            if os.path.getsize(file_path) < 1000:
                return False
            data, sr = sf.read(file_path)
            if len(data) < (sr * 0.3):
                return False
            fft = np.abs(np.fft.rfft(data))
            total_energy = np.sum(fft**2)
            if total_energy == 0:
                return False
            sorted_energies = np.sort(fft**2)
            top2_energy = np.sum(sorted_energies[-2:])
            ratio = top2_energy / total_energy
            # True speech has widely distributed frequency spectrum (ratio < 0.30)
            return bool(ratio < 0.30)
        except Exception:
            return False

    def synthesize(
        self,
        text: str,
        lang: str = "en",
        voice: str | None = None,
        persona: str = "lead",
    ) -> str:
        """
        Synthesizes speech for the given text in English, Tamil, or Hindi.
        Returns the path to the generated .wav audio file.
        """
        if not text or not text.strip():
            return ""

        clean_text = text.strip()
        lang_key = self._normalize_lang(lang)
        voices = self.VOICE_REGISTRY.get(lang_key, self.VOICE_REGISTRY["en"])
        selected_voice = voice or voices.get(
            persona, voices.get("lead", "en-IN-PrabhatNeural")
        )

        cached_file = self._get_cache_path(clean_text, selected_voice, lang_key)

        # Return cached audio ONLY if valid neural speech (rejects corrupt/tone files)
        if self.is_valid_speech_audio(cached_file):
            return cached_file

        # Speed scaling for live broadcast rhythm: Tamil & Hindi benefit from +10% rate
        rate_str = "+10%" if lang_key in ["ta", "hi"] else "+5%"
        t0 = time.time()

        # 1. Primary Engine: Edge-TTS Neural broadcast voice (Cloud API, 0MB RAM)
        if self._synthesize_edge(
            clean_text, selected_voice, cached_file, rate=rate_str
        ):
            if self.is_valid_speech_audio(cached_file):
                MetricsManager.record_tts_latency(persona, time.time() - t0)
                return cached_file

        # 2. English Fallback: Hugging Face Kokoro-82M API (Cloud API, 0MB RAM)
        if lang_key == "en":
            hf_voice = voices.get(f"hf_{persona}", "am_adam")
            if self._synthesize_hf_kokoro(clean_text, hf_voice, cached_file):
                if self.is_valid_speech_audio(cached_file):
                    MetricsManager.record_tts_latency(persona, time.time() - t0)
                    return cached_file

        # 3. Multilingual Secondary Fallback: gTTS (Cloud API, 0MB RAM)
        if self._synthesize_gtts(clean_text, lang_key, cached_file):
            if self.is_valid_speech_audio(cached_file):
                MetricsManager.record_tts_latency(persona, time.time() - t0)
                return cached_file

        # 4. Fallback Tone generator (Emergency only)
        sample_rate = 24000
        duration = max(1.0, len(clean_text) * 0.05)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        freq = 300.0 if persona == "analyst" else 450.0
        audio_data = 0.1 * np.sin(2 * np.pi * freq * t)
        fade_samples = int(sample_rate * 0.05)
        if len(audio_data) > fade_samples * 2:
            audio_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        sf.write(cached_file, audio_data, sample_rate)
        return cached_file

    def synthesize_lead(self, text: str, lang: str = "en") -> str:
        """Synthesizes Lead Play-by-Play commentary audio in specified language."""
        return self.synthesize(text, lang=lang, persona="lead")

    def synthesize_analyst(self, text: str, lang: str = "en") -> str:
        """Synthesizes Color Analyst commentary audio in specified language."""
        return self.synthesize(text, lang=lang, persona="analyst")

    def synthesize_dual(
        self, lead_text: str, analyst_text: str, lang: str = "en"
    ) -> tuple[str, float]:
        """
        Synthesizes both Lead and Analyst voices and merges them with a 0.3s pause.
        Returns a tuple of (merged_audio_path, total_duration_seconds).
        """
        lang_key = self._normalize_lang(lang)
        if not lead_text and not analyst_text:
            return "", 0.0

        if lead_text and not analyst_text:
            path = self.synthesize_lead(lead_text, lang=lang_key)
            return path, self.get_audio_duration(path)

        if analyst_text and not lead_text:
            path = self.synthesize_analyst(analyst_text, lang=lang_key)
            return path, self.get_audio_duration(path)

        hash_key = hashlib.md5(
            f"dual:{lang_key}:{lead_text}:{analyst_text}".encode()
        ).hexdigest()
        dual_file = os.path.join(self.cache_dir, f"dual_{lang_key}_{hash_key}.wav")

        if self.is_valid_speech_audio(dual_file):
            return dual_file, self.get_audio_duration(dual_file)

        lead_path = self.synthesize_lead(lead_text, lang=lang_key)
        analyst_path = self.synthesize_analyst(analyst_text, lang=lang_key)

        try:
            lead_data, sr1 = sf.read(lead_path)
            analyst_data, sr2 = sf.read(analyst_path)

            target_sr = sr1 or 24000
            # 0.3s natural broadcast pause
            pause_samples = int(target_sr * 0.3)
            pause_gap = np.zeros(
                pause_samples,
                dtype=lead_data.dtype if hasattr(lead_data, "dtype") else np.float32,
            )

            if len(lead_data.shape) > 1:
                lead_data = lead_data[:, 0]
            if len(analyst_data.shape) > 1:
                analyst_data = analyst_data[:, 0]

            merged_data = np.concatenate([lead_data, pause_gap, analyst_data])
            sf.write(dual_file, merged_data, target_sr)
            total_dur = len(merged_data) / float(target_sr)
            return dual_file, round(total_dur, 2)
        except Exception as e:
            print(f"[TTSEngine] Dual merge notice: {e}. Falling back to lead audio.")
            return lead_path, self.get_audio_duration(lead_path)

    def get_audio_duration(self, file_path: str) -> float:
        """Returns audio file duration in seconds."""
        if not file_path or not os.path.exists(file_path):
            return 0.0
        try:
            info = sf.info(file_path)
            return round(info.duration, 2)
        except Exception:
            return 5.0


if __name__ == "__main__":
    tts = TTSEngine()
    # Test English
    audio_en, dur_en = tts.synthesize_dual(
        "Livingstone smashes it for four!",
        "Terrific timing against the spin.",
        lang="en",
    )
    print(
        "EN Audio:", audio_en, "Duration:", dur_en, "Size:", os.path.getsize(audio_en)
    )
    # Test Tamil
    audio_ta, dur_ta = tts.synthesize_dual(
        "லிவிங்ஸ்டோன் அபாரமான பவுண்டரியை அடிக்கிறார்!",
        "சுழற்பந்து வீச்சை அருமையாக கையாண்டார்.",
        lang="ta",
    )
    print(
        "TA Audio:", audio_ta, "Duration:", dur_ta, "Size:", os.path.getsize(audio_ta)
    )
    # Test Hindi
    audio_hi, dur_hi = tts.synthesize_dual(
        "लिविंगस्टोन ने कवर्स पर शानदार चौका जड़ा!", "स्पिन के खिलाफ बेहतरीन फुटवर्क।", lang="hi"
    )
    print(
        "HI Audio:", audio_hi, "Duration:", dur_hi, "Size:", os.path.getsize(audio_hi)
    )
