"""Step 2b - foreground segmentation.

(a) classical  : background-difference -> Otsu threshold -> morphology -> largest blob,
                 with GrabCut as a fallback when Otsu gives an implausible mask.
(b) rembg      : U2-Net deep model (pre-trained). We use it as *pseudo* ground truth,
                 NOT as true ground truth (ABO small images come without masks).
"""
import os

import cv2
import numpy as np

# ---------------------------------------------------------------- classical
def _border_colour(img):
    """Median colour of the 1-pixel image frame ~= background colour (usually white)."""
    b = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return np.median(b, axis=0)


def _clean(mask):
    """Morphology: opening removes specks, closing fills gaps; keep largest blob; fill holes."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    mask = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, cnts, -1, 255, cv2.FILLED)        # fill interior holes
    return mask


def otsu_mask(img):
    """Distance of every pixel from the background colour, then Otsu picks the threshold."""
    diff = np.abs(img.astype(np.int16) - _border_colour(img).astype(np.int16)).max(axis=2).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    _, m = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean(m)


def grabcut_mask(img, iters=4):
    """GrabCut initialised with a rectangle just inside the image border."""
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), m, (3, 3, w - 6, h - 6), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return _clean(np.where((m == 1) | (m == 3), 255, 0).astype(np.uint8))


def classical_mask(img):
    """Otsu first; if the mask looks wrong (too small / floods the image) fall back to GrabCut.
    Returns (mask uint8 0/255, method name)."""
    m = otsu_mask(img)
    frac = (m > 0).mean()
    if 0.08 < frac < 0.90:
        return m, "otsu"
    return grabcut_mask(img), "grabcut"


# ---------------------------------------------------------------- deep (rembg)
_session = None

def rembg_mask(img):
    """U2-Net salient-object mask via the `rembg` package (lazy import: heavy dependency)."""
    global _session
    from rembg import new_session, remove
    if _session is None:
        _session = new_session(os.environ.get("REMBG_MODEL", "u2net"))   # "u2net" (176MB) on Colab; "u2netp" (4MB) for quick local tests
    a = np.array(remove(img, session=_session, only_mask=True))
    return ((a > 127) * 255).astype(np.uint8)


# ---------------------------------------------------------------- evaluation / cutouts
def iou(a, b):
    """Intersection-over-Union of two binary masks."""
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def cutout(img, mask, size=256):
    """RGBA cut-out: soft alpha edge (1px blur), cropped to the object box, longest side = size."""
    alpha = cv2.GaussianBlur(mask, (0, 0), 0.8)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([img, alpha])[y0:y1, x0:x1]
    s = size / max(rgba.shape[:2])
    return cv2.resize(rgba, (max(1, round(rgba.shape[1] * s)), max(1, round(rgba.shape[0] * s))), interpolation=cv2.INTER_AREA)
