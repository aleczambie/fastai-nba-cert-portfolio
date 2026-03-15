from __future__ import annotations
from pathlib import Path
import requests

BASE = "https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main"
FILES = [
    "regular_season_totals_2010_2024.csv",
    "regular_season_box_scores_2010_2024_part_1.csv",
    "regular_season_box_scores_2010_2024_part_2.csv",
    "regular_season_box_scores_2010_2024_part_3.csv",
]

def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def main() -> None:
    data_dir = Path("data")
    for name in FILES:
        url = f"{BASE}/{name}"
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"✓ exists: {dest}")
            continue
        print(f"↓ downloading: {name}")
        download(url, dest)
        print(f"✓ saved: {dest} ({dest.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()