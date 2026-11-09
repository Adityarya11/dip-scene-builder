"""Post-hoc cleanup of an existing artifacts/ folder: remove fabric/material SWATCH listings
(titles like "Khaki Swatch, Stone & Beam") - they are not products, but ABO files them under chair/sofa.
Rows are dropped consistently from meta.csv, embeddings, all feature matrices and the cut-out folder.

   python -m src.clean --art artifacts
Afterwards re-run:  python -m src.palette --art artifacts   and   python -m src.eval ...
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd


def clean(art):
    art = Path(art)
    meta = pd.read_csv(art / "meta.csv")
    bad = meta.caption.str.contains("swatch", case=False).values
    keep = ~bad
    for name in ["embeddings", "feat_hsv", "feat_hog", "feat_lbp"]:
        f = art / f"{name}.npy"
        if f.exists():
            X = np.load(f)
            if len(X) == len(meta):                       # idempotent: skip if already cleaned
                np.save(f, X[keep])
    for iid in meta.image_id[bad]:
        (art / "cutouts" / f"{iid}.webp").unlink(missing_ok=True)
    meta = meta[keep].reset_index(drop=True)
    meta["idx"] = range(len(meta))
    meta.to_csv(art / "meta.csv", index=False)
    Path("results").mkdir(exist_ok=True)
    json.dump({"n": len(meta), "mean_iou": float(meta.iou_classical_vs_rembg.mean()),
               "mean_iou_per_category": meta.groupby("category").iou_classical_vs_rembg.mean().round(4).to_dict(),
               "grabcut_fallback_fraction": float((meta.classical_method != "otsu").mean()),
               "note": "rembg (U2-Net) is PSEUDO ground truth, not human-annotated"}, open("results/iou.json", "w"), indent=2)
    print(f"removed {bad.sum()} swatches, {len(meta)} items left")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--art", default="artifacts")
    clean(ap.parse_args().art)
