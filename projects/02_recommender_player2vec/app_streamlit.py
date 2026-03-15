from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL = Path("models/player_similarity_profiles.joblib")

st.set_page_config(page_title="NBA Similar Players (Stats)", layout="centered")
st.title("🏀 NBA Similar Players — Performance Profile (All Seasons Combined)")

if not MODEL.exists():
    st.error("Missing profiles. Run: python projects/02_recommender_player2vec/train.py")
    st.stop()

payload = joblib.load(MODEL)
profiles: pd.DataFrame = payload["profiles"]
Xz: np.ndarray = payload["Xz"]
features: list[str] = payload["features"]

# Cosine similarity helpers
def cosine_sim_matrix(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    # X and v are already z-scored; cosine needs L2 normalization
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    vn = v / (np.linalg.norm(v) + 1e-9)
    return Xn @ vn

st.caption(
    "Similarity is computed from normalized per-36 stats + shooting rates + TS% (cosine similarity). "
    "No minutes/rotation embeddings here — this is performance style."
)

query = st.text_input("Search player name", value="Curry")
matches = profiles[profiles["playerName"].str.contains(query, case=False, na=False)].copy()
matches = matches.sort_values("total_min", ascending=False).head(50)

if matches.empty:
    st.info("No matches. Try another spelling (e.g., 'Jokic', 'Doncic', 'LeBron').")
    st.stop()

label_options = (matches["playerName"] + "  |  mins=" + matches["total_min"].astype(int).astype(str)).tolist()
choice_label = st.selectbox("Pick a player", label_options)

choice_name = choice_label.split("  |  mins=")[0]
idx = profiles.index[profiles["playerName"] == choice_name][0]
row = profiles.loc[idx]

topk = st.slider("Top-K similar players", 5, 30, 10)

# Optional: let user weight “shooter vs passer vs big” via presets
preset = st.selectbox(
    "Preset emphasis (optional)",
    ["Balanced", "Shooters", "Playmakers", "Bigs/Rebounders"],
)

weights = np.ones(len(features), dtype=np.float32)
feat_to_i = {f:i for i,f in enumerate(features)}
if preset == "Shooters":
    for f in ["FG3A_per36","FG3_PCT","TS_PCT","FGA_per36","FT_PCT"]:
        if f in feat_to_i: weights[feat_to_i[f]] = 1.8
elif preset == "Playmakers":
    for f in ["AST_per36","TOV_per36"]:
        if f in feat_to_i: weights[feat_to_i[f]] = 1.8
elif preset == "Bigs/Rebounders":
    for f in ["REB_per36","BLK_per36","FTA_per36"]:
        if f in feat_to_i: weights[feat_to_i[f]] = 1.8

if st.button("Find similar"):
    v = Xz[profiles.index.get_loc(idx)].copy()
    v = v * weights

    sims = cosine_sim_matrix(Xz * weights, v)
    sims[profiles.index.get_loc(idx)] = -1  # exclude self

    best_pos = np.argsort(-sims)[:topk]
    best = profiles.iloc[best_pos].copy()
    best["similarity"] = sims[best_pos]

    st.subheader(f"Similar to: **{row['playerName']}**")
    st.write(
        f"Profile snapshot: "
        f"PTS/36={row['PTS_per36']:.1f}, AST/36={row['AST_per36']:.1f}, REB/36={row['REB_per36']:.1f}, "
        f"3PA/36={row['FG3A_per36']:.1f}, 3P%={row['FG3_PCT']:.3f}, TS%={row['TS_PCT']:.3f}"
    )

    show_cols = ["playerName", "similarity", "PTS_per36", "AST_per36", "REB_per36", "FG3A_per36", "FG3_PCT", "TS_PCT", "games", "total_min"]
    st.dataframe(best[show_cols].reset_index(drop=True))

    # “Why” explanation: show biggest feature diffs vs query (absolute)
    st.subheader("Why these are similar (closest feature differences)")
    base_vec = profiles.loc[idx, features].to_numpy(dtype=np.float32)
    for i in range(min(5, len(best))):
        other = best.iloc[i]
        other_vec = other[features].to_numpy(dtype=np.float32)
        diffs = np.abs(other_vec - base_vec)
        top = np.argsort(diffs)[:4]  # smallest diffs = most similar features
        explain = ", ".join([f"{features[j]} Δ={diffs[j]:.2f}" for j in top])
        st.write(f"- **{other['playerName']}**: {explain}")