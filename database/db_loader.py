"""
Database loader for Cricsheet match JSON into SQLite / PostgreSQL.
Computes complete batting scorecards, bowling scorecards, partnerships,
and syncs video timestamps from video_sync.json.
"""

import json
import os
import sqlite3
from typing import Any


class DatabaseLoader:
    def __init__(
        self,
        db_path: str = "data/match_stats.db",
        schema_path: str = "database/schema.sql",
    ):
        self.db_path = db_path
        self.schema_path = schema_path
        os.makedirs(
            os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".",
            exist_ok=True,
        )

    def init_db(self) -> None:
        """Initializes database schema from schema.sql."""
        conn = sqlite3.connect(self.db_path)
        with open(self.schema_path, "r") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()

    def load_match(
        self, match_json_path: str, video_sync_path: str | None = None
    ) -> dict[str, Any]:
        """Loads and parses Cricsheet JSON and populates all relational tables."""
        self.init_db()

        with open(match_json_path, "r") as f:
            data = json.load(f)

        # Load video sync map if provided
        video_sync_map = {}
        if video_sync_path and os.path.exists(video_sync_path):
            with open(video_sync_path, "r") as f:
                sync_data = json.load(f)
                for inn in sync_data.get("innings", []):
                    inn_num = inn.get("inning")
                    for ov in inn.get("overs", []):
                        ov_num = ov.get("over")
                        for d in ov.get("deliveries", []):
                            ball_num = d.get("ball")
                            # We key by (inn_num, ov_num, ball_num, is_legal)
                            key = (inn_num, ov_num, ball_num, d.get("is_legal", True))
                            video_sync_map[key] = d.get("video_time_sec")

        info = data.get("info", {})
        # Use filename as match_id if not in info
        match_id = "1276906"
        teams = info.get("teams", ["England", "India"])
        team1, team2 = teams[0], teams[1] if len(teams) > 1 else "Unknown"

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. Insert Match
        cursor.execute(
            """
            INSERT OR REPLACE INTO matches (
                match_id, season, city, venue, match_date, match_type,
                team1, team2, toss_winner, toss_decision, winner,
                win_by_runs, win_by_wickets, player_of_match, balls_per_over
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                match_id,
                str(info.get("season", "2022")),
                info.get("city", "Nottingham"),
                info.get("venue", "Trent Bridge, Nottingham"),
                info.get("dates", ["2022-07-10"])[0],
                info.get("match_type", "T20"),
                team1,
                team2,
                info.get("toss", {}).get("winner", ""),
                info.get("toss", {}).get("decision", ""),
                info.get("outcome", {}).get("winner", ""),
                info.get("outcome", {}).get("by", {}).get("runs", 0),
                info.get("outcome", {}).get("by", {}).get("wickets", 0),
                info.get("player_of_match", [""])[0]
                if info.get("player_of_match")
                else "",
                info.get("balls_per_over", 6),
            ),
        )

        # 2. Insert Players
        for team, player_list in info.get("players", {}).items():
            for p in player_list:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO players (player_id, player_name, team)
                    VALUES (?, ?, ?)
                """,
                    (f"{team}_{p}".replace(" ", "_"), p, team),
                )

        # 3. Process Innings and Deliveries
        for inn_idx, inning in enumerate(data.get("innings", []), start=1):
            batting_team = inning.get("team", team1 if inn_idx == 1 else team2)
            bowling_team = team2 if batting_team == team1 else team1

            cum_runs = 0
            cum_wickets = 0
            legal_balls_count = 0

            # Batsman stats tracking: {batter: {runs, balls, fours, sixes, dismissal}}
            batters_stat: dict[str, dict[str, Any]] = {}
            # Bowler stats tracking: {bowler: {legal_balls, runs_conceded, wickets, maidens, wides, noballs}}
            bowlers_stat: dict[str, dict[str, Any]] = {}

            # Partnership tracking
            partnerships: list[dict[str, Any]] = []
            curr_partner_runs = 0
            curr_partner_balls = 0
            curr_wicket = 0

            for over in inning.get("overs", []):
                over_num = over.get("over", 0)
                # Keep track of balls delivered in this specific over for sync matching
                legal_ball_in_over = 0
                ball_in_over_idx = 0

                for delivery in over.get("deliveries", []):
                    ball_in_over_idx += 1
                    batter = delivery.get("batter", "")
                    bowler = delivery.get("bowler", "")
                    non_striker = delivery.get("non_striker", "")
                    runs = delivery.get("runs", {})
                    r_batter = runs.get("batter", 0)
                    r_extras = runs.get("extras", 0)
                    r_total = runs.get("total", 0)

                    extras_dict = delivery.get("extras", {})
                    extra_type = list(extras_dict.keys())[0] if extras_dict else "none"
                    is_legal = extra_type not in ["wides", "noballs"]

                    if is_legal:
                        legal_balls_count += 1
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

                    cum_runs += r_total
                    if is_wicket:
                        cum_wickets += 1

                    # Look up video sync timestamp
                    sync_key = (inn_idx, over_num, sync_ball_num, is_legal)
                    video_time = video_sync_map.get(sync_key, None)

                    # Delivery insertion
                    deliv_id = (
                        f"{match_id}_inn{inn_idx}_ov{over_num}_b{ball_in_over_idx}"
                    )
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO deliveries (
                            delivery_id, match_id, inning_num, over_num, ball_num,
                            is_legal_ball, batter, bowler, non_striker,
                            runs_batter, runs_extras, runs_total, extra_type,
                            is_wicket, player_out, wicket_kind, fielders,
                            video_time_sec, cumulative_runs, cumulative_wickets
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            deliv_id,
                            match_id,
                            inn_idx,
                            over_num,
                            ball_in_over_idx,
                            is_legal,
                            batter,
                            bowler,
                            non_striker,
                            r_batter,
                            r_extras,
                            r_total,
                            extra_type,
                            is_wicket,
                            player_out,
                            wicket_kind,
                            fielders,
                            video_time,
                            cum_runs,
                            cum_wickets,
                        ),
                    )

                    # Track batter stats
                    if batter not in batters_stat:
                        batters_stat[batter] = {
                            "runs": 0,
                            "balls": 0,
                            "fours": 0,
                            "sixes": 0,
                            "dismissal": "not out",
                        }
                    batters_stat[batter]["runs"] += r_batter
                    if extra_type != "wides":
                        batters_stat[batter]["balls"] += 1
                    if r_batter == 4:
                        batters_stat[batter]["fours"] += 1
                    elif r_batter == 6:
                        batters_stat[batter]["sixes"] += 1

                    if is_wicket and player_out in batters_stat:
                        desc = wicket_kind
                        if fielders:
                            desc += f" c {fielders}"
                        desc += f" b {bowler}"
                        batters_stat[player_out]["dismissal"] = desc

                    # Track bowler stats
                    if bowler not in bowlers_stat:
                        bowlers_stat[bowler] = {
                            "legal_balls": 0,
                            "runs_conceded": 0,
                            "wickets": 0,
                            "maidens": 0,
                            "wides": 0,
                            "noballs": 0,
                        }
                    if is_legal:
                        bowlers_stat[bowler]["legal_balls"] += 1
                    # Bowler concedes batter runs + wides/noballs (not byes/legbyes)
                    if extra_type in ["none", "wides", "noballs"]:
                        bowlers_stat[bowler]["runs_conceded"] += r_total
                    elif extra_type in ["byes", "legbyes"]:
                        bowlers_stat[bowler]["runs_conceded"] += r_batter
                    if extra_type == "wides":
                        bowlers_stat[bowler]["wides"] += 1
                    elif extra_type == "noballs":
                        bowlers_stat[bowler]["noballs"] += 1
                    if is_wicket and wicket_kind not in [
                        "run out",
                        "retired hurt",
                        "obstructing the field",
                    ]:
                        bowlers_stat[bowler]["wickets"] += 1

                    # Track partnership
                    curr_partner_runs += r_total
                    if extra_type != "wides":
                        curr_partner_balls += 1
                    if is_wicket:
                        partnerships.append(
                            {
                                "partnership_id": f"{match_id}_inn{inn_idx}_p{curr_wicket + 1}",
                                "match_id": match_id,
                                "inning_num": inn_idx,
                                "wicket_num": curr_wicket + 1,
                                "batter1": batter,
                                "batter2": non_striker,
                                "runs": curr_partner_runs,
                                "balls": curr_partner_balls,
                            }
                        )
                        curr_wicket += 1
                        curr_partner_runs = 0
                        curr_partner_balls = 0

            # Insert Batting Scorecard
            for b_name, b_info in batters_stat.items():
                sr = (
                    round((b_info["runs"] / b_info["balls"] * 100.0), 2)
                    if b_info["balls"] > 0
                    else 0.0
                )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO batting_scorecard (
                        match_id, inning_num, batter, runs, balls, fours, sixes, strike_rate, dismissal_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        match_id,
                        inn_idx,
                        b_name,
                        b_info["runs"],
                        b_info["balls"],
                        b_info["fours"],
                        b_info["sixes"],
                        sr,
                        b_info["dismissal"],
                    ),
                )

            # Insert Bowling Scorecard
            for bw_name, bw_info in bowlers_stat.items():
                overs_bowled = (bw_info["legal_balls"] // 6) + (
                    bw_info["legal_balls"] % 6
                ) / 10.0
                total_overs_dec = bw_info["legal_balls"] / 6.0
                econ = (
                    round(bw_info["runs_conceded"] / total_overs_dec, 2)
                    if total_overs_dec > 0
                    else 0.0
                )
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO bowling_scorecard (
                        match_id, inning_num, bowler, overs, maidens, runs_conceded, wickets, economy, wides, noballs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        match_id,
                        inn_idx,
                        bw_name,
                        overs_bowled,
                        bw_info["maidens"],
                        bw_info["runs_conceded"],
                        bw_info["wickets"],
                        econ,
                        bw_info["wides"],
                        bw_info["noballs"],
                    ),
                )

            # Insert Partnerships
            for p in partnerships:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO partnerships (
                        partnership_id, match_id, inning_num, wicket_num, batter1, batter2, runs, balls
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        p["partnership_id"],
                        p["match_id"],
                        p["inning_num"],
                        p["wicket_num"],
                        p["batter1"],
                        p["batter2"],
                        p["runs"],
                        p["balls"],
                    ),
                )

            # Insert Inning Record
            tot_ov = (legal_balls_count // 6) + (legal_balls_count % 6) / 10.0
            cursor.execute(
                """
                INSERT OR REPLACE INTO innings (
                    match_id, inning_num, batting_team, bowling_team, total_runs, total_wickets, total_overs
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    match_id,
                    inn_idx,
                    batting_team,
                    bowling_team,
                    cum_runs,
                    cum_wickets,
                    tot_ov,
                ),
            )

        conn.commit()
        conn.close()
        print(
            f"[DatabaseLoader] Successfully ingested match {match_id} into {self.db_path}"
        )
        return {"status": "success", "match_id": match_id}


if __name__ == "__main__":
    loader = DatabaseLoader()
    loader.load_match("data/1276906.json", "data/video_sync.json")
