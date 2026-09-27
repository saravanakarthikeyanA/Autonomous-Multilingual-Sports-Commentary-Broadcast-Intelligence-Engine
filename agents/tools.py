"""
Agent Tools for Stats Retrieval, Match State Inspection, and Read-Only SQL.
Includes strict SQL query sanitization to prevent injection or mutations.
"""

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

    def query_stats_sql(self, sql_query: str) -> list[dict[str, Any]]:
        """Executes a sanitized read-only SQL query over the match stats database."""
        safe_query = sanitize_sql_query(sql_query)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(safe_query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            return [{"error": str(e)}]
        finally:
            conn.close()

    def get_player_stats(self, player_name: str) -> dict[str, Any]:
        """Fetches current match batting and bowling records for a player."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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

        return {"player": player_name, "batting": batting_rows, "bowling": bowling_rows}

    def get_current_match_summary(self, match_id: str = "1276906") -> dict[str, Any]:
        """Retrieves match info, inning scores, top scorers, and best bowlers."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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


if __name__ == "__main__":
    tools = CricketStatsTools()
    print("Match Summary:", tools.get_current_match_summary())
    print("Player Stats (Livingstone):", tools.get_player_stats("Livingstone"))
