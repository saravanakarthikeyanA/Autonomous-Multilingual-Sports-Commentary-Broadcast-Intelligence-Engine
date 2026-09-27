"""
Financial / Stock-Market Style Live Cricket Analytics & Charts.
Generates neon terminal graphs:
1. High-Frequency Live Momentum & Run Worm Chart (Area gradient, moving averages, wickets/boundaries).
2. Live Win Probability Ticker (England vs India).
3. Match Pressure & Velocity Gauges.
"""

from typing import Any

import altair as alt
import pandas as pd


def create_momentum_worm_chart(deliveries_history: list[dict[str, Any]]) -> alt.Chart:
    """Creates a high-frequency trading terminal style Run Worm & Momentum Chart."""
    if not deliveries_history:
        # Default baseline if no deliveries yet
        df = pd.DataFrame(
            [
                {
                    "over_ball": 18.0,
                    "runs": 190,
                    "event_type": "Preload",
                    "size": 30,
                    "color": "#00F0FF",
                }
            ]
        )
    else:
        rows = []
        for i, item in enumerate(deliveries_history):
            d = item.get("deliv", {})
            st = item.get("state", {})
            over_float = d.get("over", 18) + (d.get("ball", 1) / 10.0)
            cum_runs = st.get("score", 190)

            event_type = "Dot/Run"
            marker_size = 40
            color = "#00F0FF"

            if d.get("is_wicket"):
                event_type = "Wicket"
                marker_size = 140
                color = "#FF0055"
            elif d.get("runs_batter") == 6:
                event_type = "Six"
                marker_size = 120
                color = "#FFD700"
            elif d.get("runs_batter") == 4:
                event_type = "Four"
                marker_size = 90
                color = "#00E5FF"

            rows.append(
                {
                    "over_ball": over_float,
                    "runs": cum_runs,
                    "event_type": event_type,
                    "size": marker_size,
                    "color": color,
                    "label": f"{d.get('over')}.{d.get('ball')}: {d.get('runs_total')}r",
                }
            )
        df = pd.DataFrame(rows)

    # Base Area Chart with Neon Gradient Feel
    area = (
        alt.Chart(df)
        .mark_area(
            line={"color": "#00F0FF", "width": 2.5},
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(0, 240, 255, 0.4)", offset=0),
                    alt.GradientStop(color="rgba(0, 240, 255, 0.0)", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
        )
        .encode(
            x=alt.X(
                "over_ball:Q", title="Overs (Decimal)", scale=alt.Scale(zero=False)
            ),
            y=alt.Y("runs:Q", title="Cumulative Score", scale=alt.Scale(zero=False)),
        )
    )

    # Event Points (Wickets, Fours, Sixes)
    points = (
        alt.Chart(df)
        .mark_circle(opacity=0.9)
        .encode(
            x="over_ball:Q",
            y="runs:Q",
            size=alt.Size("size:Q", legend=None),
            color=alt.Color("color:N", scale=None),
            tooltip=["label:N", "runs:Q", "event_type:N"],
        )
    )

    chart = (
        (area + points)
        .properties(
            height=230,
            title=alt.TitleParams(
                text="📈 Live Momentum & Score Progression",
                color="#00F0FF",
                fontSize=13,
                anchor="start",
            ),
        )
        .configure_view(strokeOpacity=0)
        .configure(background="transparent")
        .configure_axis(
            labelColor="#A0AEC0",
            titleColor="#00F0FF",
            gridColor="#162544",
            domainColor="#162544",
        )
    )

    return chart


def calculate_win_probability(state: dict[str, Any]) -> dict[str, float]:
    """Dynamically models win probability (England vs India) like a stock pricing model."""
    inn = state.get("inning", 1)
    score = state.get("score", 0)
    wickets = state.get("wickets", 0)
    legal_balls = state.get("legal_balls", 0)
    target = state.get("target")

    if inn == 1:
        # Projected score model based on 190+ runs at over 18
        balls_left = 120 - legal_balls
        proj_score = score + (balls_left * 1.7) - (wickets * 3)
        england_win = min(88.0, max(25.0, (proj_score - 170.0) * 1.5 + 45.0))
        india_win = 100.0 - england_win
    else:
        # Chase model in 2nd inning
        runs_needed = max(0, target - score)
        balls_left = max(1, 120 - legal_balls)
        rrr = runs_needed / (balls_left / 6.0)
        wickets_in_hand = max(1, 10 - wickets)

        # Pressure formula
        india_win = min(
            92.0, max(8.0, 50.0 + (wickets_in_hand * 4.0) - ((rrr - 8.5) * 6.0))
        )
        england_win = 100.0 - india_win

    return {
        "batting_team_prob": round(england_win if inn == 1 else india_win, 1),
        "bowling_team_prob": round(india_win if inn == 1 else england_win, 1),
    }


def create_win_probability_chart(prob_history: list[dict[str, Any]]) -> alt.Chart:
    """Creates live stock-style Win Probability timeline chart."""
    if not prob_history:
        prob_history = [{"over_ball": 18.0, "england": 65.0, "india": 35.0}]

    df = pd.DataFrame(prob_history)
    df_melt = pd.melt(
        df,
        id_vars=["over_ball"],
        value_vars=["england", "india"],
        var_name="Team",
        value_name="Probability",
    )
    df_melt["Team"] = df_melt["Team"].str.capitalize()

    chart = (
        alt.Chart(df_melt)
        .mark_line(interpolate="monotone", strokeWidth=2.5)
        .encode(
            x=alt.X(
                "over_ball:Q",
                title="Match Timeline (Overs)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("Probability:Q", title="Win %", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Team:N",
                scale=alt.Scale(
                    domain=["England", "India"], range=["#00F0FF", "#FF8000"]
                ),
            ),
            tooltip=["over_ball:Q", "Team:N", "Probability:Q"],
        )
        .properties(
            height=190,
            title=alt.TitleParams(
                text="⚡ Live Win Probability Ticker",
                color="#FFD700",
                fontSize=13,
                anchor="start",
            ),
        )
        .configure_view(strokeOpacity=0)
        .configure(background="transparent")
        .configure_axis(
            labelColor="#A0AEC0",
            titleColor="#FFD700",
            gridColor="#162544",
            domainColor="#162544",
        )
        .configure_legend(labelColor="#E2E8F0", titleColor="#00F0FF")
    )

    return chart
