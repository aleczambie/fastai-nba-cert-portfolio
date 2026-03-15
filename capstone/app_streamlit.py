from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from fastai.tabular.all import load_learner

# ---- Paths ----
WL_MODEL = Path("models/tabular_wl.pkl")
WL_FEATURES = Path("data/processed/wl_features.parquet")

SIM_MODEL = Path("models/player_similarity_profiles.joblib")

st.set_page_config(page_title="NBA Scout Assistant", layout="centered")
st.title("🏀 NBA Scout Assistant (fast.ai Capstone)")

tab1, tab2 = st.tabs(["Game Picker: Win/Loss", "Similar Players (Stats)"])

# -------------------------
# Tab 1: Game Picker Win/Loss
# -------------------------
with tab1:
    st.header("Game Picker — Win/Loss Predictor (Tabular)")
    st.caption("Uses pre-game rolling averages (last 10 games) for team + opponent. No manual stat entry.")

    if not WL_FEATURES.exists():
        st.warning("Missing features file. Run: python projects/01_tabular_wl/make_features.py")
    elif not WL_MODEL.exists():
        st.warning("Missing model file. Run: python projects/01_tabular_wl/train.py")
    else:
        df = pd.read_parquet(WL_FEATURES)
        learn = load_learner(WL_MODEL)

        seasons = sorted(df["SEASON"].unique())
        season = st.selectbox("Season", seasons, index=len(seasons) - 1)

        df_s = df[df["SEASON"] == season].copy()
        team = st.selectbox("Team", sorted(df_s["TEAM"].unique()))

        df_t = df_s[df_s["TEAM"] == team].copy().sort_values("GAME_DATE")
        df_t["label"] = df_t["GAME_DATE"].dt.strftime("%Y-%m-%d") + " | " + df_t["MATCHUP"].astype(str)

        label = st.selectbox("Pick a game", df_t["label"].tolist())
        row = df_t[df_t["label"] == label].iloc[0]

        if st.button("Predict W/L"):
            pred, _, probs = learn.predict(row)
            st.subheader(f"Prediction: **{pred}**")
            st.write({learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))})
            st.caption("Actual historical result:")
            st.write(f"**{row['WL']}**")

# -------------------------
# Tab 2: Similar Players (Stats-based)
# -------------------------
with tab2:
    st.header("Similar Players — Performance Profile (All Seasons Combined)")
    st.caption("Similarity is cosine similarity over normalized per-36 stats + shooting % + TS%.")

    if not SIM_MODEL.exists():
        st.warning("Missing similarity profiles. Run: python projects/02_recommender_player2vec/train.py")
    else:
        payload = joblib.load(SIM_MODEL)
        profiles: pd.DataFrame = payload["profiles"]
        Xz: np.ndarray = payload["Xz"]
        features: list[str] = payload["features"]

        def cosine_sim_matrix(X: np.ndarray, v: np.ndarray) -> np.ndarray:
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            vn = v / (np.linalg.norm(v) + 1e-9)
            return Xn @ vn

        query = st.text_input("Search player name", value="Jokic")
        matches = profiles[profiles["playerName"].str.contains(query, case=False, na=False)].copy()
        matches = matches.sort_values("total_min", ascending=False).head(50)

        if matches.empty:
            st.info("No matches. Try another name/spelling.")
        else:
            label_options = (matches["playerName"] + "  |  mins=" + matches["total_min"].astype(int).astype(str)).tolist()
            choice_label = st.selectbox("Pick a player", label_options)

            choice_name = choice_label.split("  |  mins=")[0]
            idx = profiles.index[profiles["playerName"] == choice_name][0]

            topk = st.slider("Top-K similar players", 5, 30, 10)

            preset = st.selectbox("Preset emphasis (optional)", ["Balanced", "Shooters", "Playmakers", "Bigs/Rebounders"])
            weights = np.ones(len(features), dtype=np.float32)
            feat_to_i = {f: i for i, f in enumerate(features)}

            if preset == "Shooters":
                for f in ["FG3A_per36", "FG3_PCT", "TS_PCT", "FGA_per36", "FT_PCT"]:
                    if f in feat_to_i:
                        weights[feat_to_i[f]] = 1.8
            elif preset == "Playmakers":
                for f in ["AST_per36", "TOV_per36"]:
                    if f in feat_to_i:
                        weights[feat_to_i[f]] = 1.8
            elif preset == "Bigs/Rebounders":
                for f in ["REB_per36", "BLK_per36", "FTA_per36"]:
                    if f in feat_to_i:
                        weights[feat_to_i[f]] = 1.8

            if st.button("Find similar players"):
                v = Xz[profiles.index.get_loc(idx)].copy()
                sims = cosine_sim_matrix(Xz * weights, v * weights)
                sims[profiles.index.get_loc(idx)] = -1

                best_pos = np.argsort(-sims)[:topk]
                best = profiles.iloc[best_pos].copy()
                best["similarity"] = sims[best_pos]

                row = profiles.loc[idx]
                st.subheader(f"Similar to: **{row['playerName']}**")
                st.write(
                    f"PTS/36={row['PTS_per36']:.1f}, AST/36={row['AST_per36']:.1f}, REB/36={row['REB_per36']:.1f}, "
                    f"3PA/36={row['FG3A_per36']:.1f}, 3P%={row['FG3_PCT']:.3f}, TS%={row['TS_PCT']:.3f}"
                )

                show_cols = ["playerName", "similarity", "PTS_per36", "AST_per36", "REB_per36", "FG3A_per36", "FG3_PCT", "TS_PCT", "games", "total_min"]
                st.dataframe(best[show_cols].reset_index(drop=True))

                st.subheader("Why these are similar (closest feature differences)")
                base_vec = profiles.loc[idx, features].to_numpy(dtype=np.float32)
                for i in range(min(5, len(best))):
                    other = best.iloc[i]
                    other_vec = other[features].to_numpy(dtype=np.float32)
                    diffs = np.abs(other_vec - base_vec)
                    top = np.argsort(diffs)[:4]
                    explain = ", ".join([f"{features[j]} Δ={diffs[j]:.2f}" for j in top])
                    st.write(f"- **{other['playerName']}**: {explain}")