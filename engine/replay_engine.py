"""
Cricket Replay and Live Game-State Engine.
Preloads match state up to Inning 1 Over 18.0, then streams deliveries
synced with video timestamps from video_sync.json.
"""

import json
from typing import Any


class ReplayEngine:
    def __init__(
        self,
        match_json_path: str = "data/1276906.json",
        video_sync_path: str = "data/video_sync.json",
    ):
        self.match_json_path = match_json_path
        self.video_sync_path = video_sync_path
        self.match_data: dict[str, Any] = {}
        self.video_sync: dict[str, Any] = {}
        self.stream_deliveries: list[dict[str, Any]] = []
        self.preload_deliveries: list[dict[str, Any]] = []
        self.current_idx: int = 0

        # Live Game State Container
        self.state: dict[str, Any] = {}
        self._load_data()
        self.reset_to_live_start()

    def _load_data(self) -> None:
        """Loads and indexes match json and video sync timeline."""
        with open(self.match_json_path, "r") as f:
            self.match_data = json.load(f)

        with open(self.video_sync_path, "r") as f:
            self.video_sync = json.load(f)

        # Build lookup set for synced deliveries: (inning, over, ball, is_legal) -> video_time_sec
        synced_lookup = {}
        for inn in self.video_sync.get("innings", []):
            inn_num = inn.get("inning")
            for ov in inn.get("overs", []):
                ov_num = ov.get("over")
                for d in ov.get("deliveries", []):
                    key = (inn_num, ov_num, d.get("ball"), d.get("is_legal", True))
                    synced_lookup[key] = d.get("video_time_sec")

        # Partition match deliveries into preload vs live stream
        innings = self.match_data.get("innings", [])
        for inn_idx, inning in enumerate(innings, start=1):
            team_name = inning.get("team", f"Team {inn_idx}")
            legal_ball_in_over = 0

            for over in inning.get("overs", []):
                ov_num = over.get("over", 0)
                legal_ball_in_over = 0

                for b_idx, delivery in enumerate(over.get("deliveries", []), start=1):
                    runs = delivery.get("runs", {})
                    extras_dict = delivery.get("extras", {})
                    extra_type = list(extras_dict.keys())[0] if extras_dict else "none"
                    is_legal = extra_type not in ["wides", "noballs"]

                    if is_legal:
                        legal_ball_in_over += 1
                        sync_ball_num = legal_ball_in_over
                    else:
                        sync_ball_num = legal_ball_in_over + 1

                    wickets = delivery.get("wickets", [])
                    is_wicket = len(wickets) > 0
                    player_out = wickets[0].get("player_out", "") if is_wicket else ""
                    wicket_kind = wickets[0].get("kind", "") if is_wicket else ""
                    fielders = (
                        ", ".join(
                            [f.get("name", "") for f in wickets[0].get("fielders", [])]
                        )
                        if is_wicket
                        else ""
                    )

                    sync_key = (inn_idx, ov_num, sync_ball_num, is_legal)
                    video_time = synced_lookup.get(sync_key, None)

                    deliv_record = {
                        "inning": inn_idx,
                        "team": team_name,
                        "over": ov_num,
                        "ball": sync_ball_num,
                        "delivery_in_over": b_idx,
                        "is_legal": is_legal,
                        "batter": delivery.get("batter", ""),
                        "bowler": delivery.get("bowler", ""),
                        "non_striker": delivery.get("non_striker", ""),
                        "runs_batter": runs.get("batter", 0),
                        "runs_extras": runs.get("extras", 0),
                        "runs_total": runs.get("total", 0),
                        "extra_type": extra_type,
                        "is_wicket": is_wicket,
                        "player_out": player_out,
                        "wicket_kind": wicket_kind,
                        "fielders": fielders,
                        "video_time_sec": video_time,
                    }

                    # Determine if this delivery is in preload or live stream
                    # Live stream covers Innings 1 Overs 18-19 and Innings 2 Overs 0-2 (present in video_sync)
                    if video_time is not None:
                        self.stream_deliveries.append(deliv_record)
                    elif inn_idx == 1 and ov_num < 18:
                        self.preload_deliveries.append(deliv_record)

    def _init_empty_state(self) -> dict[str, Any]:
        """Initializes empty state dictionary."""
        info = self.match_data.get("info", {})
        teams = info.get("teams", ["England", "India"])
        return {
            "match_info": {
                "match_id": "1276906",
                "team1": teams[0],
                "team2": teams[1],
                "venue": info.get("venue", "Trent Bridge, Nottingham"),
                "toss": info.get("toss", {}),
            },
            "inning": 1,
            "batting_team": teams[0],
            "bowling_team": teams[1],
            "score": 0,
            "wickets": 0,
            "legal_balls": 0,
            "overs_display": "0.0",
            "current_over": 0,
            "target": None,
            "crr": 0.0,
            "rrr": None,
            "striker": "",
            "non_striker": "",
            "bowler": "",
            "batters": {},  # {name: {runs, balls, fours, sixes, dismissal}}
            "bowlers": {},  # {name: {legal_balls, runs, wickets, maidens, wides, noballs}}
            "partnership": {"runs": 0, "balls": 0, "batter1": "", "batter2": ""},
            "recent_deliveries": [],
            "last_event": None,
        }

    def _apply_delivery(self, state: dict[str, Any], d: dict[str, Any]) -> None:
        """Updates the game state with a delivery event."""
        inn = d["inning"]
        if state["inning"] != inn:
            # Inning change
            state["inning"] = inn
            state["target"] = state["score"] + 1 if inn == 2 else None
            state["batting_team"] = d["team"]
            state["bowling_team"] = (
                state["match_info"]["team1"]
                if d["team"] == state["match_info"]["team2"]
                else state["match_info"]["team2"]
            )
            state["score"] = 0
            state["wickets"] = 0
            state["legal_balls"] = 0
            state["overs_display"] = "0.0"
            state["current_over"] = d["over"]
            state["partnership"] = {
                "runs": 0,
                "balls": 0,
                "batter1": d["batter"],
                "batter2": d["non_striker"],
            }

        state["striker"] = d["batter"]
        state["non_striker"] = d["non_striker"]
        state["bowler"] = d["bowler"]
        state["current_over"] = d["over"]

        # Update score & legal balls
        state["score"] += d["runs_total"]
        if d["is_legal"]:
            state["legal_balls"] += 1

        ov_int = state["legal_balls"] // 6
        b_rem = state["legal_balls"] % 6
        state["overs_display"] = f"{ov_int}.{b_rem}"

        # Update run rates
        total_overs_dec = state["legal_balls"] / 6.0
        state["crr"] = (
            round(state["score"] / total_overs_dec, 2) if total_overs_dec > 0 else 0.0
        )
        if state["target"]:
            balls_left = 120 - state["legal_balls"]
            runs_needed = state["target"] - state["score"]
            state["rrr"] = (
                round((runs_needed / (balls_left / 6.0)), 2) if balls_left > 0 else 0.0
            )

        # Update batter stats
        b_name = d["batter"]
        if b_name not in state["batters"]:
            state["batters"][b_name] = {
                "runs": 0,
                "balls": 0,
                "fours": 0,
                "sixes": 0,
                "dismissal": "not out",
            }
        state["batters"][b_name]["runs"] += d["runs_batter"]
        if d["extra_type"] != "wides":
            state["batters"][b_name]["balls"] += 1
        if d["runs_batter"] == 4:
            state["batters"][b_name]["fours"] += 1
        elif d["runs_batter"] == 6:
            state["batters"][b_name]["sixes"] += 1

        # Update bowler stats
        bw_name = d["bowler"]
        if bw_name not in state["bowlers"]:
            state["bowlers"][bw_name] = {
                "legal_balls": 0,
                "runs": 0,
                "wickets": 0,
                "maidens": 0,
                "wides": 0,
                "noballs": 0,
            }
        if d["is_legal"]:
            state["bowlers"][bw_name]["legal_balls"] += 1
        if d["extra_type"] in ["none", "wides", "noballs"]:
            state["bowlers"][bw_name]["runs"] += d["runs_total"]
        elif d["extra_type"] in ["byes", "legbyes"]:
            state["bowlers"][bw_name]["runs"] += d["runs_batter"]
        if d["extra_type"] == "wides":
            state["bowlers"][bw_name]["wides"] += 1
        elif d["extra_type"] == "noballs":
            state["bowlers"][bw_name]["noballs"] += 1

        # Update partnership
        state["partnership"]["runs"] += d["runs_total"]
        if d["extra_type"] != "wides":
            state["partnership"]["balls"] += 1
        state["partnership"]["batter1"] = d["batter"]
        state["partnership"]["batter2"] = d["non_striker"]

        # Handle Wickets
        if d["is_wicket"]:
            state["wickets"] += 1
            p_out = d["player_out"] or d["batter"]
            if p_out in state["batters"]:
                desc = d["wicket_kind"]
                if d["fielders"]:
                    desc += f" c {d['fielders']}"
                desc += f" b {d['bowler']}"
                state["batters"][p_out]["dismissal"] = desc

            if d["wicket_kind"] not in ["run out", "retired hurt"]:
                state["bowlers"][bw_name]["wickets"] += 1

            # Reset partnership
            state["partnership"] = {"runs": 0, "balls": 0, "batter1": "", "batter2": ""}

        # Recent ball summary
        short_desc = f"{d['runs_total']}"
        if d["is_wicket"]:
            short_desc = "W"
        elif d["extra_type"] == "wides":
            short_desc = f"{d['runs_total']}Wd"
        elif d["extra_type"] == "noballs":
            short_desc = f"{d['runs_total']}Nb"
        elif d["runs_batter"] == 4:
            short_desc = "4"
        elif d["runs_batter"] == 6:
            short_desc = "6"

        state["recent_deliveries"].append(
            {
                "over_ball": f"{d['over']}.{d['ball']}",
                "desc": short_desc,
                "runs": d["runs_total"],
                "is_wicket": d["is_wicket"],
                "batter": d["batter"],
                "bowler": d["bowler"],
            }
        )
        if len(state["recent_deliveries"]) > 12:
            state["recent_deliveries"].pop(0)

        state["last_event"] = d

    def reset_to_live_start(self) -> None:
        """Preloads Innings 1 (overs 0-17) and prepares the live stream from Over 18."""
        self.state = self._init_empty_state()
        for d in self.preload_deliveries:
            self._apply_delivery(self.state, d)
        self.current_idx = 0

    def step(self) -> dict[str, Any] | None:
        """Advances to the next live delivery in the synced stream."""
        if self.current_idx >= len(self.stream_deliveries):
            return None
        delivery = self.stream_deliveries[self.current_idx]
        self._apply_delivery(self.state, delivery)
        self.current_idx += 1
        return {
            "delivery": delivery,
            "state": self.get_state_snapshot(),
            "index": self.current_idx - 1,
            "total_deliveries": len(self.stream_deliveries),
        }

    def get_next_delivery(self) -> dict[str, Any] | None:
        """Alias for step() to advance to the next delivery."""
        return self.step()

    def get_delivery_at_timestamp(self, time_sec: float) -> dict[str, Any] | None:
        """Finds the most recent delivery occurring at or before video timestamp."""
        best_deliv = None
        best_idx = -1
        for i, d in enumerate(self.stream_deliveries):
            if d.get("video_time_sec") is not None and d["video_time_sec"] <= time_sec:
                best_deliv = d
                best_idx = i
        return best_deliv

    def get_state_snapshot(self) -> dict[str, Any]:
        """Returns deep-copied snapshot of current state."""
        return json.loads(json.dumps(self.state))


if __name__ == "__main__":
    engine = ReplayEngine()
    print(
        f"[ReplayEngine] Preload complete. Score before Over 18: {engine.state['batting_team']} {engine.state['score']}/{engine.state['wickets']} ({engine.state['overs_display']} ov)"
    )
    print(f"[ReplayEngine] Stream deliveries ready: {len(engine.stream_deliveries)}")
    first_step = engine.step()
    print(
        f"[ReplayEngine] First live ball (Over 18.1): {first_step['delivery']['bowler']} to {first_step['delivery']['batter']}: {first_step['delivery']['runs_total']} runs"
    )
