-- Autonomous Sports Commentary System - Schema DDL
-- Compatible with AWS RDS PostgreSQL and Local SQLite

-- 1. Matches Table
CREATE TABLE IF NOT EXISTS matches (
    match_id VARCHAR(50) PRIMARY KEY,
    season VARCHAR(20),
    city VARCHAR(100),
    venue VARCHAR(255),
    match_date DATE,
    match_type VARCHAR(20),
    team1 VARCHAR(100),
    team2 VARCHAR(100),
    toss_winner VARCHAR(100),
    toss_decision VARCHAR(20),
    winner VARCHAR(100),
    win_by_runs INT,
    win_by_wickets INT,
    player_of_match VARCHAR(100),
    balls_per_over INT DEFAULT 6
);

-- 2. Innings Table
CREATE TABLE IF NOT EXISTS innings (
    match_id VARCHAR(50),
    inning_num INT,
    batting_team VARCHAR(100),
    bowling_team VARCHAR(100),
    total_runs INT DEFAULT 0,
    total_wickets INT DEFAULT 0,
    total_overs REAL DEFAULT 0.0,
    PRIMARY KEY (match_id, inning_num),
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

-- 3. Deliveries Table
CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id VARCHAR(100) PRIMARY KEY,
    match_id VARCHAR(50),
    inning_num INT,
    over_num INT,
    ball_num INT,
    is_legal_ball BOOLEAN DEFAULT 1,
    batter VARCHAR(100),
    bowler VARCHAR(100),
    non_striker VARCHAR(100),
    runs_batter INT DEFAULT 0,
    runs_extras INT DEFAULT 0,
    runs_total INT DEFAULT 0,
    extra_type VARCHAR(50),
    is_wicket BOOLEAN DEFAULT 0,
    player_out VARCHAR(100),
    wicket_kind VARCHAR(50),
    fielders VARCHAR(255),
    video_time_sec REAL,
    cumulative_runs INT DEFAULT 0,
    cumulative_wickets INT DEFAULT 0,
    FOREIGN KEY (match_id, inning_num) REFERENCES innings(match_id, inning_num)
);

-- 4. Players Table
CREATE TABLE IF NOT EXISTS players (
    player_id VARCHAR(50) PRIMARY KEY,
    player_name VARCHAR(100),
    team VARCHAR(100)
);

-- 5. Batting Scorecard Table
CREATE TABLE IF NOT EXISTS batting_scorecard (
    match_id VARCHAR(50),
    inning_num INT,
    batter VARCHAR(100),
    runs INT DEFAULT 0,
    balls INT DEFAULT 0,
    fours INT DEFAULT 0,
    sixes INT DEFAULT 0,
    strike_rate REAL DEFAULT 0.0,
    dismissal_text VARCHAR(255),
    PRIMARY KEY (match_id, inning_num, batter)
);

-- 6. Bowling Scorecard Table
CREATE TABLE IF NOT EXISTS bowling_scorecard (
    match_id VARCHAR(50),
    inning_num INT,
    bowler VARCHAR(100),
    overs REAL DEFAULT 0.0,
    maidens INT DEFAULT 0,
    runs_conceded INT DEFAULT 0,
    wickets INT DEFAULT 0,
    economy REAL DEFAULT 0.0,
    wides INT DEFAULT 0,
    noballs INT DEFAULT 0,
    PRIMARY KEY (match_id, inning_num, bowler)
);

-- 7. Partnerships Table
CREATE TABLE IF NOT EXISTS partnerships (
    partnership_id VARCHAR(100) PRIMARY KEY,
    match_id VARCHAR(50),
    inning_num INT,
    wicket_num INT,
    batter1 VARCHAR(100),
    batter2 VARCHAR(100),
    runs INT DEFAULT 0,
    balls INT DEFAULT 0
);

-- 8. Commentary Logs Table
CREATE TABLE IF NOT EXISTS commentary_logs (
    log_id VARCHAR(100) PRIMARY KEY,
    match_id VARCHAR(50),
    inning_num INT,
    over_num INT,
    ball_num INT,
    lead_commentary TEXT,
    analyst_commentary TEXT,
    lead_audio_path VARCHAR(255),
    analyst_audio_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for ultra-fast query execution
CREATE INDEX IF NOT EXISTS idx_deliv_match_inn ON deliveries(match_id, inning_num);
CREATE INDEX IF NOT EXISTS idx_deliv_match_inn_over ON deliveries(match_id, inning_num, over_num);
CREATE INDEX IF NOT EXISTS idx_deliv_batter ON deliveries(batter);
CREATE INDEX IF NOT EXISTS idx_deliv_bowler ON deliveries(bowler);
