"""Run the whole offline pipeline: raw image -> enhance -> segment (2 ways) -> cutout -> features.

   python -m src.build --data data_test --out artifacts_test      # 48-image smoke test
   python -m src.build --data data --out artifacts                # full run (use Colab/Kaggle GPU for CLIP+rembg)

Outputs in --out:  meta.csv (+iou, method), cutouts/*.webp, feat_{hsv,hog,lbp}.npy, embeddings.npy (CLIP)
Resumable: cutouts / masks that already exist are not recomputed.
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features import HANDCRAFTED, clip_images, load_clip
from src.preprocess import enhance, load_rgb
from src.segment import classical_mask, cutout, iou, rembg_mask


def main(data, out, device, skip_clip=False):
    data, out = Path(data), Path(out)
    (out / "cutouts").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    meta = pd.read_csv(data / "meta.csv")
    imgs, feats, ious, methods = [], {k: [] for k in HANDCRAFTED}, [], []

    for r in tqdm(meta.itertuples(), total=len(meta), desc="enhance+segment+features"):
        img = enhance(load_rgb(data / r.path_local))
        imgs.append(img)
        cm, method = classical_mask(img)
        rp = out / "masks" / f"{r.image_id}_rembg.png"
        if rp.exists():
            rm = cv2.imread(str(rp), 0)
        else:
            rm = rembg_mask(img)
            cv2.imwrite(str(rp), rm)
        cv2.imwrite(str(out / "masks" / f"{r.image_id}_classical.png"), cm)
        ious.append(iou(cm, rm))
        methods.append(method)

        # final cut-out = rembg mask (better); fall back to classical if rembg returned (almost) nothing
        mask = rm if (rm > 0).mean() > 0.03 else cm
        if r.category == "wall_art":
            # framed pictures are rectangles; U2-Net "salient object" cuts out the motif inside the frame.
            # So use the bounding rectangle of the classical mask instead.
            ys, xs = np.where(cm > 0)
            if len(ys) and (cm > 0).mean() > 0.08:
                mask = np.zeros_like(cm); mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 255
        cp = out / "cutouts" / f"{r.image_id}.webp"
        if not cp.exists():
            c = cutout(img, mask)
            cv2.imwrite(str(cp), cv2.cvtColor(c, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_WEBP_QUALITY, 90])
        for k, f in HANDCRAFTED.items():
            feats[k].append(f(img, mask) if k == "hsv" else f(img))

    for k, v in feats.items():
        np.save(out / f"feat_{k}.npy", np.stack(v))
    if not skip_clip:
        clip = load_clip(device)
        np.save(out / "embeddings.npy", clip_images(imgs, clip, device))   # CLIP image embeddings (N,512)

    meta["iou_classical_vs_rembg"] = ious
    meta["classical_method"] = methods
    meta.to_csv(out / "meta.csv", index=False)
    summary = {"n": len(meta), "mean_iou": float(np.mean(ious)),
               "mean_iou_per_category": meta.groupby("category").iou_classical_vs_rembg.mean().round(4).to_dict(),
               "grabcut_fallback_fraction": float(np.mean([m == "grabcut" for m in methods])),
               "note": "rembg (U2-Net) is PSEUDO ground truth, not human-annotated"}
    Path("results").mkdir(exist_ok=True)
    json.dump(summary, open("results/iou.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-clip", action="store_true")
    a = ap.parse_args()
    main(a.data, a.out, a.device, a.skip_clip)
