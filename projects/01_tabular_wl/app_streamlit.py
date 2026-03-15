from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st
from fastai.tabular.all import load_learner

MODEL = Path("models/tabular_wl.pkl")
FEATURES = Path("data/processed/wl_features.parquet")

st.set_page_config(page_title="NBA Game Picker W/L", layout="centered")
st.title("🏀 NBA Game Picker: Win/Loss Predictor (fastai tabular)")

if not FEATURES.exists():
    st.error("Features not found. Run: python projects/01_tabular_wl/make_features.py")
    st.stop()
if not MODEL.exists():
    st.error("Model not found. Run: python projects/01_tabular_wl/train.py")
    st.stop()

df = pd.read_parquet(FEATURES)
learn = load_learner(MODEL)

# Simple selectors
season = st.selectbox("Season", sorted(df["SEASON"].unique()), index=len(df["SEASON"].unique())-1)

df_s = df[df["SEASON"] == season].copy()
team = st.selectbox("Team", sorted(df_s["TEAM"].unique()))

df_t = df_s[df_s["TEAM"] == team].copy()
df_t = df_t.sort_values("GAME_DATE")

# Show game options as "date vs/opponent"
df_t["label"] = df_t["GAME_DATE"].dt.strftime("%Y-%m-%d") + " | " + df_t["MATCHUP"].astype(str)

label = st.selectbox("Pick a game", df_t["label"].tolist())

row = df_t[df_t["label"] == label].iloc[0]

st.write("This app auto-loads pre-game features (rolling averages) — no manual stat entry.")

if st.button("Predict"):
    # learner.predict expects a row-like object with feature columns
    pred, _, probs = learn.predict(row)
    st.subheader(f"Prediction: **{pred}**")
    st.write({learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))})

    st.caption("Actual result (for this historical game):")
    st.write(f"**{row['WL']}**")