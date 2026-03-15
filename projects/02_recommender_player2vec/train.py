from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

DATA_DIR = Path("data")
OUT = Path("models")
OUT.mkdir(exist_ok=True)

BOX = [
    DATA_DIR / "regular_season_box_scores_2010_2024_part_1.csv",
    DATA_DIR / "regular_season_box_scores_2010_2024_part_2.csv",
    DATA_DIR / "regular_season_box_scores_2010_2024_part_3.csv",
]

MODEL_OUT = OUT / "player_similarity_profiles.joblib"


def minutes_to_float(x) -> float | None:
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    if ":" in s:
        try:
            mm, ss = s.split(":")
            return float(mm) + float(ss) / 60.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def pick_col(cols: list[str], *candidates: str) -> str:
    """Return the first candidate that exists in cols, else raise."""
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"Could not find any of {candidates} in columns.")


def load_box_scores(max_rows: int | None = None) -> pd.DataFrame:
    dfs = []
    for p in BOX:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}  (run: python scripts/download_data.py)")
        dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)

    cols = list(df.columns)

    # Detect likely column names (supports many schemas)
    game = pick_col(cols, "gameId", "GAME_ID", "Game_ID", "GAMEID")
    pid = pick_col(cols, "personId", "PLAYER_ID", "playerId", "PERSON_ID")
    name = pick_col(cols, "personName", "PLAYER_NAME", "playerName", "PLAYER")

    mins = pick_col(cols, "minutes", "MIN", "min", "MINUTES")

    pts = pick_col(cols, "points", "PTS", "pts")
    ast = pick_col(cols, "assists", "AST", "ast")
    reb = None
    for cand in ("reboundsTotal", "REB", "reb", "TOT_REB"):
        if cand in cols:
            reb = cand
            break
    if reb is None:
        # if split available:
        oreb = next((c for c in ("reboundsOffensive", "OREB") if c in cols), None)
        dreb = next((c for c in ("reboundsDefensive", "DREB") if c in cols), None)
        if oreb and dreb:
            df["REB_TMP"] = pd.to_numeric(df[oreb], errors="coerce").fillna(0) + pd.to_numeric(df[dreb], errors="coerce").fillna(0)
            reb = "REB_TMP"
        else:
            raise KeyError("Could not find REB (or OREB+DREB).")

    tov = pick_col(cols, "turnovers", "TOV", "tov")
    stl = pick_col(cols, "steals", "STL", "stl")
    blk = pick_col(cols, "blocks", "BLK", "blk")

    fgm = pick_col(cols, "fieldGoalsMade", "FGM", "fgm")
    fga = pick_col(cols, "fieldGoalsAttempted", "FGA", "fga")
    fg3m = pick_col(cols, "threePointersMade", "FG3M", "fg3m", "3PM")
    fg3a = pick_col(cols, "threePointersAttempted", "FG3A", "fg3a", "3PA")
    ftm = pick_col(cols, "freeThrowsMade", "FTM", "ftm")
    fta = pick_col(cols, "freeThrowsAttempted", "FTA", "fta")

    keep = [game, pid, name, mins, pts, ast, reb, tov, stl, blk, fgm, fga, fg3m, fg3a, ftm, fta]
    df = df[keep].copy()
    df.columns = [
        "gameId", "playerId", "playerName", "minutes",
        "PTS", "AST", "REB", "TOV", "STL", "BLK",
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA"
    ]

    df["minutes"] = df["minutes"].apply(minutes_to_float)
    df = df.dropna(subset=["minutes"])
    df = df[df["minutes"] > 0].copy()

    # numeric coercion for stats
    for c in ["PTS","AST","REB","TOV","STL","BLK","FGM","FGA","FG3M","FG3A","FTM","FTA"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    return df.reset_index(drop=True)


def build_player_profiles(df: pd.DataFrame, min_total_minutes: float = 800.0) -> pd.DataFrame:
    # Aggregate totals across all seasons (combined)
    g = df.groupby(["playerId", "playerName"], as_index=False).agg(
        games=("gameId", "nunique"),
        total_min=("minutes", "sum"),
        PTS=("PTS", "sum"),
        AST=("AST", "sum"),
        REB=("REB", "sum"),
        STL=("STL", "sum"),
        BLK=("BLK", "sum"),
        TOV=("TOV", "sum"),
        FGM=("FGM", "sum"),
        FGA=("FGA", "sum"),
        FG3M=("FG3M", "sum"),
        FG3A=("FG3A", "sum"),
        FTM=("FTM", "sum"),
        FTA=("FTA", "sum"),
    )

    # Filter out tiny samples (reduces weird results)
    g = g[g["total_min"] >= min_total_minutes].copy()

    # Per-36 rates
    denom = g["total_min"].replace(0, np.nan)
    for c in ["PTS","AST","REB","STL","BLK","TOV","FGA","FG3A","FTA"]:
        g[f"{c}_per36"] = (g[c] / denom) * 36.0

    # Shooting percentages (from totals)
    g["FG_PCT"] = np.where(g["FGA"] > 0, g["FGM"] / g["FGA"], np.nan)
    g["FG3_PCT"] = np.where(g["FG3A"] > 0, g["FG3M"] / g["FG3A"], np.nan)
    g["FT_PCT"] = np.where(g["FTA"] > 0, g["FTM"] / g["FTA"], np.nan)

    # True Shooting %
    ts_denom = 2.0 * (g["FGA"] + 0.44 * g["FTA"])
    g["TS_PCT"] = np.where(ts_denom > 0, g["PTS"] / ts_denom, np.nan)

    # Keep a clean feature set for similarity
    feature_cols = [
        "PTS_per36","AST_per36","REB_per36","STL_per36","BLK_per36","TOV_per36",
        "FGA_per36","FG3A_per36","FTA_per36",
        "FG_PCT","FG3_PCT","FT_PCT","TS_PCT",
    ]

    # Fill any NaNs in percentages (rare after filtering) with column medians
    for c in feature_cols:
        if g[c].isna().any():
            g[c] = g[c].fillna(g[c].median())

    return g, feature_cols


def zscore_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, dict]:
    X = df[cols].to_numpy(dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    Xz = (X - mean) / std
    scaler = {"mean": mean, "std": std, "cols": cols}
    return Xz, scaler


def main() -> None:
    print("Loading box scores...")
    df = load_box_scores()
    print("Loaded rows:", len(df))
    if len(df) == 0:
        raise RuntimeError("Box score dataframe is empty. Check CSV columns/schema.")

    print("Building player profiles (all seasons combined)...")
    profiles, feature_cols = build_player_profiles(df, min_total_minutes=800.0)
    print(f"Players kept after filtering: {len(profiles)}")

    Xz, scaler = zscore_matrix(profiles, feature_cols)

    payload = {
        "profiles": profiles,       # includes playerId, playerName, games, totals, per36, pct
        "features": feature_cols,
        "Xz": Xz,                   # normalized feature matrix (same row order as profiles)
        "scaler": scaler,
    }

    joblib.dump(payload, MODEL_OUT)
    print(f"Saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()