"""Step 4a - colour palettes and a simple colour-harmony score.

* dominant_colors : k-means (k chosen in 3..5 by silhouette) on the object's pixels in CIELAB.
                    LAB is perceptual, so Euclidean k-means distances ~ perceived colour differences.
* harmony_score   : hue/saturation compatibility of a SET of items (1 = harmonious, 0 = clashing).
                    Rules from basic colour theory: analogous, complementary, triadic hues look good;
                    neutrals (white/grey/black/beige-ish, low chroma) match anything.
* python -m src.palette : precompute palettes for every cut-out -> artifacts/palettes.npz
"""
import argparse
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

KMAX = 5
NEUTRAL_CHROMA = 18.0          # LAB chroma below this = "neutral" (grey/white/black)


def rgb2lab(rgb):
    """(n,3) uint8 RGB -> (n,3) float LAB (L 0..100, a/b ~ -110..110)."""
    lab = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    return lab * np.array([100 / 255, 1, 1], np.float32) - np.array([0, 128, 128], np.float32)


def lab2rgb(lab):
    """(n,3) float LAB -> (n,3) uint8 RGB (for drawing swatches)."""
    x = (np.asarray(lab, np.float32) + np.array([0, 128, 128], np.float32)) * np.array([255 / 100, 1, 1], np.float32)
    return cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)


def dominant_colors(rgba, ks=(3, 4, 5), seed=0):
    """K-means in LAB over opaque pixels. Returns (lab (5,3), weights (5,)) padded with zeros."""
    px = rgba[rgba[..., 3] > 128][:, :3]
    if len(px) < 30:                                   # degenerate cut-out
        px = rgba[..., :3].reshape(-1, 3)
    rng = np.random.default_rng(seed)
    px = px[rng.choice(len(px), min(len(px), 2000), replace=False)]
    lab = rgb2lab(px)
    best = None
    for k in ks:
        if len(np.unique(lab.round(0), axis=0)) < k:
            continue
        km = KMeans(k, n_init=3, random_state=seed).fit(lab)
        sc = silhouette_score(lab, km.labels_, sample_size=500, random_state=seed)
        if best is None or sc > best[0]:
            best = (sc, km)
    if best is None:                                   # flat-colour image
        c, w = lab.mean(0, keepdims=True), np.array([1.0])
    else:
        km = best[1]
        c, w = km.cluster_centers_, np.bincount(km.labels_, minlength=len(km.cluster_centers_)) / len(lab)
    order = np.argsort(-w)
    out_c, out_w = np.zeros((KMAX, 3), np.float32), np.zeros(KMAX, np.float32)
    out_c[:len(c)], out_w[:len(c)] = c[order], w[order]
    return out_c, out_w


# ------------------------------------------------------------------ harmony
def chroma_hue(lab):
    """LAB (a,b) -> chroma and hue angle in degrees."""
    return np.hypot(lab[..., 1], lab[..., 2]), np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


def item_summary(lab, w):
    """Reduce an item's palette to (hue, chroma, is_neutral): the weighted-mean colour of its *chromatic* part."""
    ch, hue = chroma_hue(lab)
    m = (w > 0) & (ch >= NEUTRAL_CHROMA)
    if m.sum() == 0 or w[m].sum() < 0.15:              # < 15% of the object is coloured -> neutral item
        return 0.0, float((ch * w).sum() / max(w.sum(), 1e-6)), True
    ww = w[m] / w[m].sum()
    ang = np.radians(hue[m])
    h = np.degrees(np.arctan2((ww * np.sin(ang)).sum(), (ww * np.cos(ang)).sum())) % 360   # circular mean
    return float(h), float((ww * ch[m]).sum()), False


def pair_score(a, b):
    """Compatibility of two item summaries in [0,1] = 0.7*hue + 0.3*saturation(chroma) agreement."""
    (ha, ca, na), (hb, cb, nb) = a, b
    if na or nb:
        return 0.95                                    # neutrals go with everything
    d = abs(ha - hb); d = min(d, 360 - d)              # hue distance 0..180
    hue = max(1 - d / 45,                              # analogous (same family)
              0.9 * max(0, 1 - abs(d - 180) / 30),     # complementary
              0.7 * max(0, 1 - abs(d - 120) / 25))     # triadic
    sat = max(0.0, 1 - abs(ca - cb) / 60)              # similar vividness
    return 0.7 * hue + 0.3 * sat


def harmony_score(summaries):
    """Mean pair score over all item pairs of a scene."""
    pairs = list(combinations(summaries, 2))
    return float(np.mean([pair_score(a, b) for a, b in pairs])) if pairs else 1.0


def hue_spread(labs, ws):
    """Second, independent-ish measure: circular standard deviation (degrees) of ALL chromatic palette
    colours of the scene (weighted). Lower = colours sit in a narrow hue band = more coherent."""
    hs, wt = [], []
    for lab, w in zip(labs, ws):
        ch, hue = chroma_hue(lab)
        m = (w > 0) & (ch >= NEUTRAL_CHROMA)
        hs += list(hue[m]); wt += list(w[m] * ch[m])
    if not hs:
        return 0.0
    a, wt = np.radians(hs), np.array(wt)
    R = np.hypot((wt * np.sin(a)).sum(), (wt * np.cos(a)).sum()) / wt.sum()
    return float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-6)))))


# ------------------------------------------------------------------ precompute
def build(art):
    art = Path(art)
    import pandas as pd
    meta = pd.read_csv(art / "meta.csv")
    labs, ws = [], []
    for iid in meta.image_id:
        c = cv2.cvtColor(cv2.imread(str(art / "cutouts" / f"{iid}.webp"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        l, w = dominant_colors(c); labs.append(l); ws.append(w)
    np.savez_compressed(art / "palettes.npz", lab=np.stack(labs), w=np.stack(ws))
    print("palettes", np.stack(labs).shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--art", default="artifacts")
    build(ap.parse_args().art)
