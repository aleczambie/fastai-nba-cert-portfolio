# Project 02: Performance Similarity (Stats-Based)

Goal: Find "similar players" based on performance profile:
- per-36 scoring + playmaking + rebounding + defense proxies
- shooting rates (FG%, 3P%, FT%)
- efficiency (TS%)

Data source:
- regular_season_box_scores_2010_2024_part_1/2/3.csv (downloaded via scripts/download_data.py)

Method:
1) Aggregate all seasons combined into one player profile
2) Compute per-36 features + percentages + TS%
3) Normalize features (z-score)
4) Similarity = cosine similarity between normalized vectors

Run:
- Build profiles: `python projects/02_recommender_player2vec/train.py`
- App: `streamlit run projects/02_recommender_player2vec/app_streamlit.py`