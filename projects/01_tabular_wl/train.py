from __future__ import annotations

from pathlib import Path
import pandas as pd
from fastai.tabular.all import *

FEATURES = Path("data/processed/wl_features.parquet")
OUT = Path("models")
OUT.mkdir(exist_ok=True)

def main() -> None:
    if not FEATURES.exists():
        raise FileNotFoundError(
            "Features not found. Run: python projects/01_tabular_wl/make_features.py"
        )

    df = pd.read_parquet(FEATURES)

    # Categorical + continuous
    cat_names = ["TEAM", "OPP_TEAM", "SEASON", "is_home"]
    # all rolling columns are continuous
    cont_names = [c for c in df.columns if c.startswith("roll_") or c.startswith("opp_roll_")]
    y_name = "WL"

    # Keep IDs for app filtering (not used as features)
    # We'll exclude these from cont/cat lists by not listing them.
    procs = [Categorify, FillMissing, Normalize]

    splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(df))
    to = TabularPandas(
        df,
        procs=procs,
        cat_names=cat_names,
        cont_names=cont_names,
        y_names=y_name,
        splits=splits
    )

    dls = to.dataloaders(bs=1024)
    learn = tabular_learner(dls, metrics=[accuracy, RocAucBinary()])
    learn.fit_one_cycle(6, 1e-2)

    model_path = OUT / "tabular_wl.pkl"
    learn.export(model_path)
    print(f"Saved model: {model_path}")

if __name__ == "__main__":
    main()