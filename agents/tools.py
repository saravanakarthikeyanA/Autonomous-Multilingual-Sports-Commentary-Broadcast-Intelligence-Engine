"""
Agent Tools for Stats Retrieval, Match State Inspection, and Read-Only SQL.
Includes strict SQL query sanitization to prevent injection or mutations.
"""

import os
import re
import sqlite3
from typing import Any

from agents.config import AgentConfig


def sanitize_sql_query(query: str) -> str:
    """Ensures SQL queries are strictly read-only and safe."""
    cleaned = query.strip()
    # Strip markdown code blocks if present
    cleaned = re.sub(r"^```(sql)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned)
    cleaned = cleaned.strip()

    # Block destructive keywords
    disallowed = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "truncate",
        "exec",
        "replace",
    ]
    tokens = re.findall(r"\b\w+\b", cleaned.lower())
    for token in tokens:
        if token in disallowed:
            raise ValueError(
                f"Security Policy Violation: Unsafe SQL operation '{token}' is blocked."
            )

    if not cleaned.lower().startswith("select") and not cleaned.lower().startswith(
        "with"
    ):
        raise ValueError(
            "Security Policy Violation: Only SELECT/WITH read queries are permitted."
        )

    return cleaned


class CricketStatsTools:
    def __init__(self, db_path: str = AgentConfig.DB_PATH):
        self.db_path = db_path
        self._ensure_db_initialized()

    def _ensure_db_initialized(self) -> None:
        """Verifies database exists and has tables, auto-populating if missing."""
        import os

        need_init = False
        if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
            need_init = True
        else:
            try:
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='matches';"
                )
                row = cur.fetchone()
                conn.close()
                if not row:
                    need_init = True
            except (sqlite3.Error, OSError):
                need_init = True

        if need_init:
            try:
                from database.db_loader import DatabaseLoader

                loader = DatabaseLoader(db_path=self.db_path)
                loader.load_match("data/1276906.json", "data/video_sync.json")
            except Exception as e:  # noqa: BLE001
                print(f"[CricketStatsTools] Auto-init notice: {e}")

    def _get_read_only_connection(self) -> sqlite3.Connection:
        """Returns a strict read-only SQLite connection enforced at the driver level."""
        abs_path = os.path.abspath(self.db_path)
        try:
            conn = sqlite3.connect(f"file:{abs_path}?mode=ro", uri=True)
        except sqlite3.Error:
            conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def query_stats_sql(self, sql_query: str) -> list[dict[str, Any]]:
        """Executes a sanitized read-only SQL query over the match stats database."""
        self._ensure_db_initialized()
        safe_query = sanitize_sql_query(sql_query)
        try:
            conn = self._get_read_only_connection()
            cursor = conn.cursor()
            cursor.execute(safe_query)
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:  # noqa: BLE001
            return [{"error": str(e)}]

    def get_player_stats(self, player_name: str) -> dict[str, Any]:
        """Fetches current match batting and bowling records for a player."""
        self._ensure_db_initialized()
        try:
            conn = self._get_read_only_connection()
            cursor = conn.cursor()

            # Batting stats
            cursor.execute(
                """
                SELECT batter, runs, balls, fours, sixes, strike_rate, dismissal_text
                FROM batting_scorecard
                WHERE batter LIKE ? OR batter = ?
            """,
                (f"%{player_name}%", player_name),
            )
            batting_rows = [dict(r) for r in cursor.fetchall()]

            # Bowling stats
            cursor.execute(
                """
                SELECT bowler, overs, maidens, runs_conceded, wickets, economy
                FROM bowling_scorecard
                WHERE bowler LIKE ? OR bowler = ?
            """,
                (f"%{player_name}%", player_name),
            )
            bowling_rows = [dict(r) for r in cursor.fetchall()]
            conn.close()
            return {
                "player": player_name,
                "batting": batting_rows,
                "bowling": bowling_rows,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "player": player_name,
                "batting": [],
                "bowling": [],
                "error": str(e),
            }

    def get_current_match_summary(self, match_id: str = "1276906") -> dict[str, Any]:
        """Retrieves match info, inning scores, top scorers, and best bowlers."""
        self._ensure_db_initialized()
        try:
            conn = self._get_read_only_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
            match_info = dict(cursor.fetchone() or {})

            cursor.execute(
                "SELECT * FROM innings WHERE match_id = ? ORDER BY inning_num ASC",
                (match_id,),
            )
            innings = [dict(r) for r in cursor.fetchall()]

            cursor.execute(
                """
                SELECT batter, runs, balls, strike_rate, inning_num
                FROM batting_scorecard
                WHERE match_id = ?
                ORDER BY runs DESC LIMIT 5
            """,
                (match_id,),
            )
            top_batters = [dict(r) for r in cursor.fetchall()]

            cursor.execute(
                """
                SELECT bowler, overs, runs_conceded, wickets, economy, inning_num
                FROM bowling_scorecard
                WHERE match_id = ?
                ORDER BY wickets DESC, economy ASC LIMIT 5
            """,
                (match_id,),
            )
            top_bowlers = [dict(r) for r in cursor.fetchall()]
            conn.close()

            return {
                "match": match_info,
                "innings": innings,
                "top_batters": top_batters,
                "top_bowlers": top_bowlers,
            }
        except Exception as e:  # noqa: BLE001
            return {
                "match": {"match_id": match_id},
                "innings": [],
                "top_batters": [],
                "top_bowlers": [],
                "error": str(e),
            }


if __name__ == "__main__":
    tools = CricketStatsTools()
    print("Match Summary:", tools.get_current_match_summary())
    print("Player Stats (Livingstone):", tools.get_player_stats("Livingstone"))
