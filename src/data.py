"""Step 1 - download ABO listing metadata + small product images and build meta.csv.

Only two things are fetched from the public ABO bucket (anonymous HTTP):
  * listings/metadata/listings_*.json.gz   (product type, English title, image id)
  * images/metadata/images.csv.gz          (image id -> relative path of images/small/...)
  * images/small/<path>                    (the ~100px product pictures we actually use)
Everything is cached on disk, so re-running the script only downloads what is missing.

Usage:  python -m src.data --per-cat 190          # full (~1500 images)
        python -m src.data --per-cat 6 --out data_test   # 50-image smoke test
"""
import argparse, gzip, json, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"
SHARDS = list("0123456789abcdef")            # listings_0 ... listings_f

# ABO product_type -> our category name (these are the 8 "slots" the composer uses)
CATEGORY_MAP = {
    "SOFA": "sofa",
    "CHAIR": "chair", "BEAN_BAG_CHAIR": "chair",
    "TABLE": "table", "DESK": "table",
    "LAMP": "lamp", "HOME_LIGHTING_AND_LAMPS": "lamp",
    "RUG": "rug",
    "PLANTER": "plant_vase", "VASE": "plant_vase",
    "SHELF": "shelf", "CABINET": "shelf", "DRESSER": "shelf",
    "WALL_ART": "wall_art",
}


def fetch(url, dest: Path):
    """Download url -> dest unless it already exists (resumable)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return False
        dest.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def english_title(entry):
    """Return the English item_name of a listing (or None)."""
    for t in entry.get("item_name", []):
        if t.get("language_tag", "").startswith("en"):
            return t["value"]
    return None


def load_candidates(raw: Path):
    """Read all listing shards -> DataFrame of (item_id, image_id, category, caption)."""
    for s in SHARDS:
        fetch(f"{BASE}/listings/metadata/listings_{s}.json.gz", raw / f"listings_{s}.json.gz")
    rows = []
    for s in SHARDS:
        with gzip.open(raw / f"listings_{s}.json.gz", "rt", encoding="utf8") as f:
            for line in f:
                e = json.loads(line)
                cat = CATEGORY_MAP.get(e.get("product_type", [{}])[0].get("value"))
                cap = english_title(e)
                # skip fabric/material swatch listings (not products, but ABO files them under chair/sofa)
                if cat and cap and e.get("main_image_id") and "swatch" not in cap.lower():
                    rows.append((e["item_id"], e["main_image_id"], cat, cap))
    return pd.DataFrame(rows, columns=["item_id", "image_id", "category", "caption"]).drop_duplicates("image_id")


def build(per_cat, out: Path, raw: Path = Path("data/raw"), seed=0):
    out.mkdir(parents=True, exist_ok=True)
    fetch(f"{BASE}/images/metadata/images.csv.gz", raw / "images.csv.gz")
    paths = pd.read_csv(raw / "images.csv.gz")[["image_id", "path"]]
    cand = load_candidates(raw).merge(paths, on="image_id")

    # deterministic shuffle -> small test subset is always a prefix of the full subset
    picked = (cand.sample(frac=1, random_state=seed).groupby("category").head(per_cat)
              .sort_values(["category", "image_id"]).reset_index(drop=True))
    picked["file"] = "images/" + picked["image_id"] + ".jpg"

    jobs = [(f"{BASE}/images/small/{p}", out / f) for p, f in zip(picked["path"], picked["file"])]
    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(lambda j: fetch(*j), jobs))
    picked = picked[ok].reset_index(drop=True)             # keep only images that really downloaded
    picked.insert(0, "idx", range(len(picked)))
    picked = picked.rename(columns={"file": "path_local"})
    picked[["idx", "image_id", "category", "caption", "path_local", "path"]].to_csv(out / "meta.csv", index=False)
    print(picked.category.value_counts().to_string(), f"\ntotal={len(picked)}  failed={len(ok) - sum(ok)}")
    return picked


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=190)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    build(a.per_cat, Path(a.out))
