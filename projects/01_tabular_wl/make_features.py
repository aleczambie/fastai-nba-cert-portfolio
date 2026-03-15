from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW = Path("data/regular_season_totals_2010_2024.csv")
OUTDIR = Path("data/processed")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT = OUTDIR / "wl_features.parquet"


def _col(df: pd.DataFrame, *names: str) -> str:
    """Return the first existing column name from candidates."""
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of these columns exist: {names}")


def make_features(n_rolling: int = 10) -> pd.DataFrame:
    df = pd.read_csv(RAW)

    game_id = _col(df, "GAME_ID", "gameId", "Game_ID")
    game_date = _col(df, "GAME_DATE", "GAME_DATE_EST", "GAME_DATE_UTC", "gameDate")
    team_abbr = _col(df, "TEAM_ABBREVIATION", "TEAM", "teamTricode")
    season = _col(df, "SEASON_YEAR", "SEASON", "season")
    matchup = _col(df, "MATCHUP", "matchup")
    wl = _col(df, "WL", "wl")

    # Use only available games if present
    if "AVAILABLE_FLAG" in df.columns:
        df = df[df["AVAILABLE_FLAG"] == 1].copy()

    df = df[df[wl].isin(["W", "L"])].copy()

    # Parse date
    df[game_date] = pd.to_datetime(df[game_date], errors="coerce")
    df = df.dropna(subset=[game_date]).copy()

    # Home/away from matchup string
    df["is_home"] = df[matchup].astype(str).str.contains("vs\\.").astype(int)

    # Pick numeric stat columns to roll (use what exists)
    candidate_stats = [
        "FGM","FGA","FG_PCT","FG3M","FG3A","FG3_PCT","FTM","FTA","FT_PCT",
        "OREB","DREB","REB","AST","TOV","STL","BLK","PF","PTS","PLUS_MINUS"
    ]
    stat_cols = [c for c in candidate_stats if c in df.columns]

    # Sort by team chronology
    df = df.sort_values([team_abbr, game_date, game_id]).copy()

    # Rolling means from previous N games (shift to ensure "pre-game")
    for c in stat_cols:
        df[f"roll_{n_rolling}_{c}"] = (
            df.groupby(team_abbr)[c]
              .transform(lambda s: s.shift(1).rolling(n_rolling, min_periods=3).mean())
        )

    # Drop early rows with missing rolling window
    roll_cols = [f"roll_{n_rolling}_{c}" for c in stat_cols]
    df = df.dropna(subset=roll_cols).copy()

    # Opponent rolling features: each GAME_ID has 2 rows (team + opponent).
    # Merge the "other row" rolling features as opp_* columns.
    base_cols = [game_id, team_abbr, season, game_date, matchup, "is_home", wl] + roll_cols
    feat = df[base_cols].copy()

    opp = feat[[game_id, team_abbr] + roll_cols].copy()
    opp = opp.rename(columns={team_abbr: "OPP_TEAM"})
    opp = opp.rename(columns={c: "opp_" + c for c in roll_cols})

    merged = feat.merge(opp, on=game_id, how="inner")

    # Keep rows where opponent team is different
    merged = merged[merged["OPP_TEAM"] != merged[team_abbr]].copy()

    # Final dataset
    merged = merged.rename(columns={
        team_abbr: "TEAM",
        season: "SEASON",
        matchup: "MATCHUP",
        wl: "WL",
        game_date: "GAME_DATE",
        game_id: "GAME_ID",
    })

    return merged


def main() -> None:
    df = make_features(n_rolling=10)
    df.to_parquet(OUT, index=False)
    print(f"Saved features: {OUT}  rows={len(df):,}  cols={df.shape[1]}")

if __name__ == "__main__":
    main()