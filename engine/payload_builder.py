"""
Pure Match Event Payload Builder.
Builds the 31-delivery match timeline payload linking delivery events to video timestamps
without any pre-baked commentary or pre-generated audio.
100% dynamic live generation happens at runtime via LangGraph multi-agent LLM and TTS.
"""

from typing import Any

from engine.replay_engine import ReplayEngine


def build_stream_sync_payload(
    engine: Any = None, tts: Any = None, lang: str = "en", *args, **kwargs
) -> list[dict[str, Any]]:
    """
    Builds the raw match timeline payload for all stream deliveries.
    Contains ONLY delivery event metadata and video timestamps.
    """
    cum_engine = ReplayEngine()
    cum_engine.reset_to_live_start()
    payload = []

    for d in cum_engine.stream_deliveries:
        step_res = cum_engine.step()
        st_snap = step_res["state"]

        payload.append(
            {
                "over": d["over"],
                "ball": d["ball"],
                "team": d["team"],
                "score": f"{st_snap['score']}/{st_snap['wickets']}",
                "overs_display": st_snap["overs_display"],
                "batter": d["batter"],
                "bowler": d["bowler"],
                "non_striker": d.get("non_striker", ""),
                "runs_batter": d.get("runs_batter", 0),
                "runs_extras": d.get("runs_extras", 0),
                "runs_total": d["runs_total"],
                "extra_type": d.get("extra_type", "none"),
                "is_wicket": d["is_wicket"],
                "player_out": d.get("player_out", ""),
                "wicket_kind": d.get("wicket_kind", ""),
                "fielders": d.get("fielders", []),
                "crr": st_snap.get("crr", "10.56"),
                "target": st_snap.get("target"),
                "rrr": st_snap.get("rrr"),
                "video_time_sec": d.get("video_time_sec"),
                "broadcast_time_sec": (d["video_time_sec"] + 8.0)
                if d.get("video_time_sec") is not None
                else None,
            }
        )

    return payload


if __name__ == "__main__":
    eng = ReplayEngine()
    res = build_stream_sync_payload(eng)
    print(
        f"Built {len(res)} raw delivery events. Sample ball: Over {res[0]['over']}.{res[0]['ball']} ({res[0]['bowler']} to {res[0]['batter']})"
    )
