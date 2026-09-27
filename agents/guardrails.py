"""
Strict Factual Grounding and Anti-Hallucination Guardrail Engine.
Validates generated commentary across English, Tamil, and Hindi against
ground-truth ball event data before broadcasting.
"""

from typing import Any


class CommentaryGuardrails:
    # Multilingual Keyword Dictionaries for Anti-Hallucination Grounding
    WICKET_KEYWORDS = [
        # English
        "out!",
        "gone!",
        "wicket!",
        "dismissed",
        "takes the wicket",
        "has to walk",
        "caught behind",
        "clean bowled",
        "is out",
        "was out",
        "trapped lbw",
        # Tamil
        "விக்கெட்!",
        "விக்கெட் விழுந்தது",
        "விக்கெட் வீழ்த்தினார்",
        "ஆட்டமிழந்தார்",
        "வெளியேறினார்",
        "கேட்ச் பிடித்தார்",
        "போல்ட் ஆனார்",
        "அவுட்!",
        # Hindi
        "विकेट!",
        "आउट",
        "पवेलियन",
        "बोल्ड कर दिया",
        "बोल्ड किया",
        "कैच आउट",
        "विकेट गिरा",
        "विकेट झटका",
        "विकेट निकाला",
        "बड़ा झटका",
    ]

    SIX_KEYWORDS = [
        # English
        "maximum",
        "into the stands",
        "over the ropes",
        "six runs",
        "hits a six",
        "huge six",
        "massive six",
        "hits it for six",
        # Tamil
        "சிக்ஸர்",
        "ஆறு ரன்கள்",
        "ஆறு ரன்",
        "வானளாவிய சிக்ஸர்",
        "கம்பீரமான சிக்ஸர்",
        # Hindi
        "छक्का",
        "छह रन",
        "गगनचुंबी छक्का",
        "दर्शक दीर्घा",
        "मैक्सिमम",
    ]

    FOUR_KEYWORDS = [
        # English
        "to the fence",
        "four runs",
        "cracks a four",
        "finds the boundary",
        "races to the boundary",
        "smashing four",
        "hits a four",
        "boundary!",
        # Tamil
        "பவுண்டரி",
        "நான்கு ரன்கள்",
        "நான்கு ரன்",
        "அபார பவுண்டரி",
        # Hindi
        "चौका",
        "चार रन",
        "बाउंड्री",
        "शानदार चौका",
    ]

    DOT_KEYWORDS = [
        # English
        "no run",
        "dot ball",
        "defends solidly for no run",
        "cannot pierce",
        # Tamil
        "ரன் இல்லை",
        "டாட் பால்",
        "ரன் எடுக்கவில்லை",
        # Hindi
        "कोई रन नहीं",
        "डॉट गेंद",
        "डॉट बॉल",
        "रन नहीं बन सका",
    ]

    @staticmethod
    def validate_delivery_commentary(
        commentary_text: str, delivery: dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Validates that generated commentary text is strictly grounded in the delivery event.
        Works across English, Tamil, and Hindi text.
        """
        text_lower = commentary_text.lower()
        runs_batter = delivery.get("runs_batter", 0)
        runs_total = delivery.get("runs_total", 0)
        is_wicket = delivery.get("is_wicket", False)
        extra_type = delivery.get("extra_type", "none")

        # 1. Wicket False Positive Check
        claimed_wicket = any(
            kw in text_lower for kw in CommentaryGuardrails.WICKET_KEYWORDS
        )
        if claimed_wicket and not is_wicket:
            return (
                False,
                f"Hallucination Detected: Commentary claims a wicket, but delivery was not a dismissal (runs: {runs_total}).",
            )

        # 2. Six / Maximum False Positive Check
        claimed_six = any(kw in text_lower for kw in CommentaryGuardrails.SIX_KEYWORDS)
        if claimed_six and runs_batter != 6:
            return (
                False,
                f"Hallucination Detected: Commentary claims a six, but batter scored {runs_batter} runs.",
            )

        # 3. Four / Boundary False Positive Check
        claimed_four = any(
            kw in text_lower for kw in CommentaryGuardrails.FOUR_KEYWORDS
        )
        if claimed_four and runs_batter != 4:
            return (
                False,
                f"Hallucination Detected: Commentary claims a four, but batter scored {runs_batter} runs.",
            )

        # 4. Dot Ball Hallucination Check
        claimed_dot = any(kw in text_lower for kw in CommentaryGuardrails.DOT_KEYWORDS)
        if claimed_dot and runs_total > 1:
            return (
                False,
                f"Hallucination Detected: Commentary claims no run / dot ball, but {runs_total} runs were scored.",
            )

        return True, "Passed factual grounding validation."


if __name__ == "__main__":
    test_delivery = {
        "batter": "LS Livingstone",
        "bowler": "HV Patel",
        "runs_batter": 1,
        "runs_total": 1,
        "is_wicket": False,
        "extra_type": "none",
    }

    good_en = "Livingstone nudges Harshal Patel into the gap for a brisk single."
    bad_en = (
        "Livingstone launches Harshal Patel high into the stands for a massive six!"
    )
    good_ta = "லிவிங்ஸ்டோன் ஒரு ரன் எடுத்து முனையை மாற்றுகிறார்."
    bad_ta = "லிவிங்ஸ்டோன் வானளாவிய சிக்ஸர் அடிக்கிறார்!"

    print(
        "Good EN check:",
        CommentaryGuardrails.validate_delivery_commentary(good_en, test_delivery),
    )
    print(
        "Bad EN check:",
        CommentaryGuardrails.validate_delivery_commentary(bad_en, test_delivery),
    )
    print(
        "Good TA check:",
        CommentaryGuardrails.validate_delivery_commentary(good_ta, test_delivery),
    )
    print(
        "Bad TA check:",
        CommentaryGuardrails.validate_delivery_commentary(bad_ta, test_delivery),
    )
