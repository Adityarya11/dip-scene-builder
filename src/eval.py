"""Step 3c - evaluation + automatic figures.

   python -m src.eval --data data --art artifacts          # everything that exists

Metrics (all saved in results/):
  * NDCG@10 per feature type (query = an item image, leave-one-out, relevance = same category)
  * CLIP text->image NDCG@10 (query = "a photo of a <category>")  [extra row, other features have no text]
  * segmentation IoU classical vs rembg (rembg = PSEUDO ground truth)
  * scene timing + palette-compatibility with harmony on/off   (added by evaluate_scenes)
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.index import Index

FIG = Path("results/figures")
COLORS = {"hsv": "#4C78A8", "hog": "#F58518", "lbp": "#54A24B", "clip": "#B279A2"}


def ndcg_at_k(rel, k=10):
    """rel = 0/1 relevance of the ranked list. Ideal list is k relevant items (>=190 exist per category)."""
    rel = np.asarray(rel[:k], float)
    disc = 1 / np.log2(np.arange(2, k + 2))
    return float((rel * disc[:len(rel)]).sum() / disc.sum())


def eval_retrieval(feats, cats, k=10):
    """Leave-one-out image->image retrieval. Returns {feature: (mean NDCG@10, per-category dict, P@10)}."""
    cats = np.asarray(cats)
    res = {}
    for name, X in feats.items():
        idx, nd = Index(X, name), []
        for i in range(len(cats)):
            ids, _ = idx.search(X[i], k, exclude=i)
            nd.append(ndcg_at_k(cats[ids] == cats[i], k))
        nd = np.array(nd)
        res[name] = (nd.mean(), {c: nd[cats == c].mean() for c in np.unique(cats)}, nd)
    return res


def eval_text(emb, cats, clip, k=10):
    """CLIP text->image: one query per category, mean NDCG@10."""
    from src.features import clip_text
    cats = np.asarray(cats)
    uc = np.unique(cats)
    q = clip_text([f"a photo of a {c.replace('_', ' ')}" for c in uc], clip)
    idx = Index(emb, "clip")
    return {c: ndcg_at_k(cats[idx.search(q[j], k)[0]] == c, k) for j, c in enumerate(uc)}


# ------------------------------------------------------------------ figures
def fig_ndcg(table):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    names = list(table)
    v = [table[n] for n in names]
    ax.bar(names, v, color=[COLORS.get(n, "#888") for n in names])
    for i, x in enumerate(v):
        ax.text(i, x + 0.01, f"{x:.2f}", ha="center")
    ax.set_ylim(0, 1.05); ax.set_ylabel("NDCG@10"); ax.set_title("Image retrieval (relevance = same category)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "fig3_ndcg.png", dpi=150); plt.close(fig)


def fig_stages(data, meta, n=4):
    """Figure 1: raw -> enhanced -> segmented -> cut-out strip for n items of different categories."""
    from src.preprocess import enhance, load_rgb
    from src.segment import classical_mask, rembg_mask, cutout
    rows = meta.groupby("category").head(1).head(n)
    fig, ax = plt.subplots(len(rows), 4, figsize=(8, 2 * len(rows)))
    for r, (_, m) in enumerate(rows.iterrows()):
        raw = load_rgb(Path(data) / m.path_local)
        enh = enhance(raw)
        mask = rembg_mask(enh)
        seg = enh.copy(); seg[mask == 0] = (seg[mask == 0] * 0.25 + np.array([255, 0, 255]) * 0.0).astype(np.uint8)
        cut = cutout(enh, mask)
        bg = np.full((*cut.shape[:2], 3), 235, np.uint8)
        a = cut[..., 3:] / 255.0
        comp = (cut[..., :3] * a + bg * (1 - a)).astype(np.uint8)
        for c, (im, t) in enumerate([(raw, "raw"), (enh, "enhanced"), (seg, "segmented"), (comp, "cut-out")]):
            ax[r, c].imshow(im); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(t)
    fig.tight_layout(); fig.savefig(FIG / "fig1_pipeline_stages.png", dpi=150); plt.close(fig)


def fig_masks(data, art, meta, n=6):
    """Figure 2: classical vs rembg mask examples (with IoU)."""
    from src.preprocess import enhance, load_rgb
    sel = meta.sort_values("iou_classical_vs_rembg").iloc[np.linspace(0, len(meta) - 1, n).astype(int)]
    fig, ax = plt.subplots(3, n, figsize=(2 * n, 6.4))
    for j, (_, m) in enumerate(sel.iterrows()):
        enh = enhance(load_rgb(Path(data) / m.path_local))
        cm = plt.imread(Path(art) / "masks" / f"{m.image_id}_classical.png")
        rm = plt.imread(Path(art) / "masks" / f"{m.image_id}_rembg.png")
        for i, (im, cmap) in enumerate([(enh, None), (cm, "gray"), (rm, "gray")]):
            ax[i, j].imshow(im, cmap=cmap); ax[i, j].axis("off")
        ax[0, j].set_title(f"IoU {m.iou_classical_vs_rembg:.2f}", fontsize=9)
    for i, t in enumerate(["image", "classical", "rembg"]):
        ax[i, 0].text(-0.08, 0.5, t, transform=ax[i, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle(f"Segmentation: mean IoU = {meta.iou_classical_vs_rembg.mean():.3f}  (rembg = pseudo ground truth)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_masks_iou.png", dpi=150); plt.close(fig)


def run(data, art, make_figs=True, clip_text_eval=True):
    art = Path(art); Path("results").mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(art / "meta.csv")
    feats = {k: np.load(art / f"feat_{k}.npy") for k in ["hsv", "hog", "lbp"]}
    if (art / "embeddings.npy").exists():
        feats["clip"] = np.load(art / "embeddings.npy")
    res = eval_retrieval(feats, meta.category)
    table = {k: v[0] for k, v in res.items()}
    df = pd.DataFrame({k: v[1] for k, v in res.items()}).T
    df.insert(0, "mean_NDCG@10", pd.Series(table))
    out = {"ndcg10_image_query": table}
    if "clip" in feats and clip_text_eval:
        from src.features import load_clip
        t = eval_text(feats["clip"], meta.category, load_clip())
        out["ndcg10_clip_text_query"] = {"mean": float(np.mean(list(t.values()))), **t}
    df.round(4).to_csv("results/ndcg_table.csv")
    out["n_items"] = len(meta)
    json.dump(out, open("results/metrics.json", "w"), indent=2, default=float)
    print(df.round(3).to_string())
    if make_figs:
        fig_ndcg(table); fig_masks(data, art, meta); fig_stages(data, meta)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--art", default="artifacts")
    a = ap.parse_args()
    run(a.data, a.art)
