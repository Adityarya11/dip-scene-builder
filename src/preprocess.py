"""Step 2a - classical Digital Image Processing pre-processing (OpenCV + NumPy only).

Order:  resize -> denoise -> gray-world white balance -> CLAHE (contrast)
Each step is its own small function so students can show before/after for every stage.
"""
import cv2
import numpy as np

SIZE = 256  # longest side after resizing


def load_rgb(path):
    """Read an image from disk as an RGB uint8 array."""
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def resize_long_side(img, size=SIZE):
    """Resize keeping aspect ratio so the longest side == size (bicubic = smooth upscaling)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)


def denoise(img, median_k=3, sigma=0.8):
    """Median filter removes JPEG salt-and-pepper blocks, light Gaussian smooths remaining noise."""
    img = cv2.medianBlur(img, median_k)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def gray_world(img, bg_thresh=235, strength=0.5):
    """Gray-world white balance: scale each channel so the *object* average becomes gray.
    Product photos have a big white background that would bias the mean, so pixels
    that are almost white (min channel > bg_thresh) are ignored when computing the gains."""
    f = img.astype(np.float32)
    fg = f.min(axis=2) < bg_thresh
    if fg.sum() < 50:                       # (nearly) all white image -> nothing to balance
        return img
    means = f[fg].mean(axis=0)              # mean R, G, B of foreground pixels
    gains = means.mean() / np.maximum(means, 1e-3)
    # Full gray-world turns e.g. beige chairs blue (the object really IS beige), so we apply
    # only half of the correction and clip it: a compromise between fixing casts and product fidelity.
    gains = np.clip(1 + strength * (gains - 1), 0.85, 1.18)
    return np.clip(f * gains, 0, 255).astype(np.uint8)


def clahe_l(img, clip=2.0, tiles=8):
    """CLAHE (adaptive histogram equalisation) on the L channel of LAB only,
    so contrast improves without changing colours."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhance(img):
    """Full pipeline on an RGB image. Returns the enhanced image."""
    return clahe_l(gray_world(denoise(resize_long_side(img))))


def stages(img):
    """Return every intermediate result (for the before/after figure)."""
    r = resize_long_side(img)
    d = denoise(r)
    w = gray_world(d)
    c = clahe_l(w)
    return {"raw": img, "resized": r, "denoised": d, "white-balanced": w, "CLAHE": c}
