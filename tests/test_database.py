"""
Unit tests for Database Schema and Ingestion.
"""

import os
import sqlite3

import pytest

from database.db_loader import DatabaseLoader


@pytest.fixture(scope="module")
def setup_db():
    test_db = "data/test_stats.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    loader = DatabaseLoader(db_path=test_db, schema_path="database/schema.sql")
    loader.load_match("data/1276906.json", "data/video_sync.json")
    yield test_db
    if os.path.exists(test_db):
        os.remove(test_db)


def test_tables_populated(setup_db):
    conn = sqlite3.connect(setup_db)
    cursor = conn.cursor()

    # Matches
    cursor.execute("SELECT COUNT(*) FROM matches")
    assert cursor.fetchone()[0] == 1

    # Deliveries
    cursor.execute("SELECT COUNT(*) FROM deliveries")
    deliv_count = cursor.fetchone()[0]
    assert deliv_count > 200

    # Scorecard
    cursor.execute("SELECT COUNT(*) FROM batting_scorecard")
    assert cursor.fetchone()[0] > 0

    # Inning 1 total score
    cursor.execute("SELECT total_runs, total_wickets FROM innings WHERE inning_num = 1")
    inn1 = cursor.fetchone()
    assert inn1[0] > 150

    conn.close()
