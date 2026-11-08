"""Step 3a - image descriptors. Three hand-crafted (classical DIP) baselines + CLIP.
Every function maps ONE RGB uint8 image -> 1-D float32 vector (L2-normalised so that
the dot product of two vectors is their cosine similarity)."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def _norm(v):
    v = v.astype(np.float32).ravel()
    return v / (np.linalg.norm(v) + 1e-8)


def hsv_hist(img, mask=None, bins=(8, 4, 4)):
    """Joint HSV colour histogram (8 hue x 4 sat x 4 val = 128-d). Optional mask ignores the background."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], mask, bins, [0, 180, 0, 256, 0, 256])
    return _norm(np.sqrt(h))           # sqrt = Hellinger-style scaling, works better with cosine


def hog_feat(img, size=128):
    """Histogram of Oriented Gradients on a 128x128 grey image (shape / edge layout)."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (size, size))
    return _norm(hog(g, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)))


def lbp_feat(img, size=128, grid=4):
    """Local Binary Pattern (uniform, P=8, R=1) histograms over a 4x4 grid of cells (texture)."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (size, size))
    lbp = local_binary_pattern(g, 8, 1, "uniform")             # values 0..9
    s, out = size // grid, []
    for i in range(grid):
        for j in range(grid):
            cell = lbp[i * s:(i + 1) * s, j * s:(j + 1) * s]
            out.append(np.bincount(cell.astype(int).ravel(), minlength=10) / cell.size)
    return _norm(np.concatenate(out))


HANDCRAFTED = {"hsv": hsv_hist, "hog": hog_feat, "lbp": lbp_feat}


# ------------------------------------------------------------------ CLIP
def load_clip(device="cpu"):
    """OpenCLIP ViT-B/32 (OpenAI weights, ~340 MB). Returns (model, preprocess, tokenizer)."""
    import open_clip, torch
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    return model.eval().to(device), pre, open_clip.get_tokenizer("ViT-B-32")


def clip_images(imgs, clip, device="cpu", bs=64):
    """Embed a list of RGB arrays -> (N,512) normalised matrix."""
    import torch
    from PIL import Image
    model, pre, _ = clip
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = torch.stack([pre(Image.fromarray(a)) for a in imgs[i:i + bs]]).to(device)
            f = model.encode_image(x)
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def clip_text(texts, clip, device="cpu"):
    """Embed a list of strings -> (N,512) normalised matrix."""
    import torch
    model, _, tok = clip
    with torch.no_grad():
        f = model.encode_text(tok(texts).to(device))
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float32)
