# Capstone: NBA Scout Assistant (Local Demo)

This capstone combines two course-aligned workflows:

## Tab 1 — Game Picker Win/Loss (Tabular)
- Uses pre-game features built from rolling averages over the previous 10 games
- Trains a fastai tabular classifier (W vs L)
- UI: select Season → Team → Game → Predict

## Tab 2 — Similar Players (Performance Profile)
- Aggregates all seasons into a per-player performance profile
- Features: per-36 stats + shooting % + TS%
- Similarity: cosine similarity in normalized feature space
- UI: search player → similar players + brief “why”

## Run
1) Download data:
   `python scripts/download_data.py`

2) Project 1:
   - `python projects/01_tabular_wl/make_features.py`
   - `python projects/01_tabular_wl/train.py`

3) Project 2:
   - `python projects/02_recommender_player2vec/train.py`

4) Capstone app:
   `streamlit run capstone/app_streamlit.py`
   