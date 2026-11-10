"""Repair stale palette rows from existing cutouts. No model training/downloads.
Run: python scripts/refresh_palettes.py
"""
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.palette import rgb2lab
import cv2
cv2.setNumThreads(1)


def main():
    art = ROOT / 'artifacts'
    meta = pd.read_csv(art / 'meta.csv')
    def extract(iid):
        p = art / 'cutouts' / f'{iid}.webp'
        if not p.exists():
            return np.zeros((5, 3), np.float32), np.zeros(5, np.float32)
        rgba = np.array(Image.open(p).convert('RGBA'))
        # Bounded-sample LAB k-means: same DIP method, no silhouette sweep.
        pixels = rgba[rgba[..., 3] > 128, :3]
        if not len(pixels):
            pixels = rgba[..., :3].reshape(-1, 3)
        pixels = pixels[np.linspace(0, len(pixels)-1, min(512, len(pixels)), dtype=int)]
        lab = rgb2lab(pixels).astype(np.float32)
        k = min(3, len(np.unique(lab, axis=0)))
        # Sorted luminance supplies deterministic initial clusters.
        labels = np.empty(len(lab), np.int32)
        labels[np.argsort(lab[:, 0], kind='stable')] = np.minimum(np.arange(len(lab))*k//len(lab), k-1)
        _, labels, centers = cv2.kmeans(lab, k, labels[:, None],
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, .2),
            1, cv2.KMEANS_USE_INITIAL_LABELS)
        w = np.bincount(labels.ravel(), minlength=k).astype(np.float32)/len(lab)
        order = np.argsort(-w)
        out_c, out_w = np.zeros((5, 3), np.float32), np.zeros(5, np.float32)
        out_c[:k], out_w[:k] = centers[order], w[order]
        return out_c, out_w
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(extract, meta.image_id))
    labs, weights = zip(*rows)
    np.savez_compressed(art / 'palettes.npz', lab=np.stack(labs), w=np.stack(weights),
                        image_id=meta.image_id.astype(str).to_numpy(dtype=str))
    print(f'Rebuilt {len(rows)} ID-keyed palettes from cached cutouts (no retraining).')


if __name__ == '__main__':
    main()
