"""Step 4b - prompt -> scene image (the "2.5D" composer).

  prompt --keywords--> slots --CLIP text search--> top-K per slot --brute-force best combo-->
  placement rules --> procedural room --> colour match (Reinhard) --> drop shadow --> alpha blend

Everything after retrieval is plain NumPy/OpenCV image processing.
"""
import re
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.index import Index
from src.palette import harmony_score, item_summary, lab2rgb, pair_score, rgb2lab

W, H = 768, 512
FLOOR_Y = 0.60                         # wall/floor boundary (fraction of H)


# ---------------------------------------------------------------- 1. prompt -> slots
# slot = category, CLIP phrase, x (canvas fraction), y (where the feet / centre are, fraction of H)
def S(cat, q, x, y=0.90):
    return dict(cat=cat, q=q, x=x, y=y)


SCENES = {
    "reading corner": [S("shelf", "bookshelf", .13, .80), S("rug", "rug", .50, .90), S("chair", "armchair", .44),
                       S("lamp", "floor lamp", .72, .84), S("plant_vase", "plant in pot", .90, .88)],
    "living room": [S("wall_art", "framed wall art", .50, .30), S("sofa", "sofa", .50, .80), S("rug", "rug", .50, .92),
                    S("table", "coffee table", .50, .95), S("lamp", "floor lamp", .86, .82),
                    S("plant_vase", "plant in pot", .12, .86)],
    "home office": [S("wall_art", "framed wall art", .45, .27), S("shelf", "bookcase", .12, .80), S("table", "desk", .45, .86),
                    S("chair", "office chair", .64, .94), S("lamp", "desk lamp", .82, .82),
                    S("plant_vase", "plant in pot", .92, .88)],
    "dining room": [S("wall_art", "framed wall art", .50, .28), S("table", "dining table", .50, .88),
                    S("chair", "dining chair", .25, .92), S("chair", "dining chair", .75, .92),
                    S("rug", "rug", .50, .93)],
    "bedroom": [S("wall_art", "framed wall art", .50, .28), S("rug", "rug", .50, .92), S("chair", "accent chair", .30),
                S("shelf", "dresser", .72, .82), S("plant_vase", "plant in pot", .90, .88)],
    "entryway": [S("wall_art", "framed wall art", .50, .30), S("shelf", "console cabinet", .50, .82),
                 S("lamp", "lamp", .22, .84), S("plant_vase", "plant in pot", .80, .86), S("rug", "rug", .50, .92)],
}
# word prefix -> scene (first match in the prompt wins, an exact scene name always wins)
ALIASES = {"reading": "reading corner", "book": "reading corner", "nook": "reading corner", "living": "living room",
           "lounge": "living room", "sofa": "living room", "office": "home office", "study": "home office",
           "work": "home office", "desk": "home office", "dining": "dining room", "kitchen": "dining room",
           "bedroom": "bedroom", "sleep": "bedroom", "entry": "entryway", "hall": "entryway"}
STOP = {"a", "an", "the", "for", "my", "in", "with", "and", "room", "corner", "area", "space", "of", "to"}


def parse_prompt(prompt):
    """Return (scene name, style words). Unknown prompt -> living room. 'cozy reading corner' -> ('reading corner','cozy')."""
    p = prompt.lower()
    words = re.findall(r"[a-z]+", p)
    scene = next((s for w in words for k, s in ALIASES.items() if w.startswith(k)), "living room")
    scene = next((k for k in SCENES if k in p), scene)
    used = set(scene.split())
    style = [w for w in words if w not in STOP and w not in used and not any(w.startswith(k) for k in ALIASES)]
    return scene, " ".join(style)


# ---------------------------------------------------------------- 2. library (precomputed artifacts)
class Library:
    """Loads meta.csv, CLIP embeddings, palettes and (lazily) cut-outs from the artifacts folder."""

    def __init__(self, art="artifacts", clip=None):
        self.art = Path(art)
        self.meta = pd.read_csv(self.art / "meta.csv")
        self.emb = np.load(self.art / "embeddings.npy")
        p = np.load(self.art / "palettes.npz")
        self.pal_lab, self.pal_w = p["lab"], p["w"]
        self.summ = [item_summary(l, w) for l, w in zip(self.pal_lab, self.pal_w)]
        self.index = Index(self.emb, "clip")
        self.cat = self.meta.category.values
        # quality filter: if the two segmenters disagree (IoU < 0.3) the cut-out is probably bad -> never use it in a scene
        # (wall art uses a rectangle mask instead, so it is exempt)
        iou = self.meta.get("iou_classical_vs_rembg", pd.Series(1.0, index=self.meta.index)).values
        self.ok = (iou >= 0.3) | (self.cat == "wall_art")
        self.clip, self._cache = clip, {}

    def encode(self, texts):
        from src.features import clip_text
        return clip_text(texts, self.clip)

    def cutout(self, i):
        if i not in self._cache:
            im = cv2.imread(str(self.art / "cutouts" / f"{self.meta.image_id[i]}.webp"), cv2.IMREAD_UNCHANGED)
            self._cache[i] = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        return self._cache[i]

    def swatch(self, i, n=5):
        """Palette of item i as RGB colours (only non-empty clusters)."""
        k = int((self.pal_w[i] > 0).sum())
        return lab2rgb(self.pal_lab[i][:min(n, k)])


# ---------------------------------------------------------------- 3. choose items
LAMBDA = 3.0        # weight of harmony vs CLIP text score (see choose_items)


def choose_items(lib, slots, style, harmony=True, K=5):
    """Top-K candidates per slot by CLIP text similarity (restricted to the slot's category),
    then brute-force every combination (K^slots <= ~15k) and keep the best.
        combo score = mean_i s_i + LAMBDA * harmony(combo)
    where s_i = (sim_i - best_sim_of_slot) / 0.05 so that CLIP differences and harmony are on similar scales.
    With harmony=False the harmony term is dropped -> simply the top-1 item of every slot."""
    qv = lib.encode([f"a photo of a {style} {s['q']}".replace("  ", " ") for s in slots])
    cands, sims = [], []
    for s, q in zip(slots, qv):
        ids, sc = lib.index.search(q, K, mask=(lib.cat == s["cat"]) & lib.ok)
        cands.append(ids)
        sims.append((sc - sc[0]) / 0.05)
    if not harmony:
        return [int(c[0]) for c in cands], cands
    # pair scores of all candidate items, computed once
    allids = sorted({int(i) for c in cands for i in c})
    pos = {i: j for j, i in enumerate(allids)}
    P = np.array([[pair_score(lib.summ[a], lib.summ[b]) for b in allids] for a in allids])
    best, best_s = None, -1e9
    for combo in product(*[range(len(c)) for c in cands]):
        ids = [int(cands[j][c]) for j, c in enumerate(combo)]
        clip_s = np.mean([sims[j][c] for j, c in enumerate(combo)])
        pp = [pos[i] for i in ids]
        hs = np.mean([P[a, b] for x, a in enumerate(pp) for b in pp[x + 1:]])
        s = clip_s + LAMBDA * hs
        if s > best_s:
            best, best_s = ids, s
    return best, cands


# ---------------------------------------------------------------- 4. procedural room
def make_room(wall_ab, seed=0):
    """Gradient wall + wood-ish floor with plank lines, baseboard and vignette. wall_ab = tint (a,b) in LAB."""
    rng = np.random.default_rng(seed)
    fy = int(FLOOR_Y * H)
    lab = np.zeros((H, W, 3), np.float32)
    lab[:fy, :, 0] = np.linspace(91, 82, fy)[:, None]                  # wall: light, slightly darker toward the floor
    lab[:fy, :, 1], lab[:fy, :, 2] = wall_ab
    lab[fy:, :, 0] = (62 - 22 * np.linspace(0, 1, H - fy))[:, None]    # floor gets darker toward the viewer
    lab[fy:, :, 1], lab[fy:, :, 2] = 10, 26
    room = cv2.cvtColor(np.clip((lab + [0, 128, 128]) * [255 / 100, 1, 1], 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    room = room.astype(np.float32)
    for k in range(-12, 28):                                           # floorboards converge to a vanishing point
        x0 = W / 2 + k * 70
        cv2.line(room, (int(W / 2 + (x0 - W / 2) * 0.25), fy), (int(x0 * 1.6 - W * 0.3), H), (120, 85, 55), 1, cv2.LINE_AA)
    room[fy - 6:fy + 2] *= 0.82                                        # baseboard shadow line
    room += rng.normal(0, 1.5, room.shape)                             # film grain
    yy, xx = np.mgrid[0:H, 0:W]
    vig = 1 - 0.11 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return np.clip(room * vig[..., None], 0, 255).astype(np.uint8)


def wall_tint(items_summ):
    """Very light wall colour with a whisper of the scene's average hue -> keeps the room harmonious."""
    ab = [(np.cos(np.radians(h)) * c, np.sin(np.radians(h)) * c) for h, c, neutral in items_summ if not neutral]
    return tuple(0.18 * np.mean(ab, axis=0)) if ab else (1.5, 7.0)


# ---------------------------------------------------------------- 5. colour match, shadow, blend
def reinhard(rgba, room, strength=0.25):
    """Reinhard colour transfer (match LAB mean/std of the object to the room), blended with the original.
    Full transfer would repaint the product, so only `strength` of it is applied."""
    rgb, a = rgba[..., :3], rgba[..., 3] > 128
    if a.sum() < 20:
        return rgba
    src = rgb2lab(rgb.reshape(-1, 3)).reshape(rgb.shape)
    tgt = rgb2lab(room.reshape(-1, 3))
    ms, ss = src[a].mean(0), src[a].std(0) + 1e-3
    mt, st = tgt.mean(0), tgt.std(0) + 1e-3
    out = (src - ms) * np.clip(st / ss, 0.5, 2.0) + mt
    out = (1 - strength) * src + strength * out
    rgb2 = lab2rgb(out.reshape(-1, 3)).reshape(rgb.shape)
    return np.dstack([rgb2, rgba[..., 3]])


def blend(canvas, rgba, x0, y0, opacity=1.0):
    """Alpha-blend an RGBA patch onto the float canvas at top-left (x0,y0), clipped at the borders."""
    h, w = rgba.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    p = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32)
    a = p[..., 3:4] / 255.0 * opacity
    canvas[Y0:Y1, X0:X1] = canvas[Y0:Y1, X0:X1] * (1 - a) + p[..., :3] * a


def drop_shadow(canvas, alpha, x0, y_base, kind):
    """Gaussian-blurred shadow. Floor objects: silhouette squashed flat on the floor at the feet.
    Wall / shelf objects: silhouette offset down-right and blurred."""
    h, w = alpha.shape
    if kind == "wall":
        patch = np.dstack([np.zeros((h, w, 3), np.uint8), cv2.GaussianBlur(alpha, (0, 0), 5)])
        blend(canvas, patch, x0 + 6, y_base - h + 8, 0.35)
    else:
        # contact shadow: an ellipse as wide as the object's "feet" (opaque columns in the lowest 12% rows)
        cols = np.where((alpha[int(h * 0.88):] > 128).any(axis=0))[0]
        fx0, fx1 = (cols.min(), cols.max()) if len(cols) else (0, w - 1)
        fw = max(8, int((fx1 - fx0) * 1.15))
        hs = max(8, int(min(fw * 0.22, h * 0.12)))
        sh = np.zeros((hs * 3, fw + 40), np.uint8)
        cv2.ellipse(sh, (sh.shape[1] // 2, sh.shape[0] // 2), (fw // 2, hs), 0, 0, 360, 255, -1)
        sh = cv2.GaussianBlur(sh, (0, 0), 5)                              # Gaussian blur = soft penumbra
        patch = np.dstack([np.zeros((*sh.shape, 3), np.uint8), sh])
        blend(canvas, patch, x0 + (fx0 + fx1) // 2 - sh.shape[1] // 2, y_base - sh.shape[0] // 2, 0.45)


# ---------------------------------------------------------------- 6. placement rules
# size priors: ('w' or 'h', fraction of canvas width/height). Aspect ratio of the cut-out is kept.
PRIORS = {"sofa": ("w", .40), "chair": ("h", .36), "table": ("w", .26), "lamp": ("h", .50), "rug": ("w", .55),
          "plant_vase": ("h", .20), "shelf": ("h", .46), "wall_art": ("w", .17)}


def place(canvas, lib, idx, slot, room):
    """Scale, colour-match, shadow and blend ONE item according to its category rules."""
    cat = slot["cat"]
    rgba = reinhard(lib.cutout(idx), room)
    if cat == "chair" and slot["x"] > 0.6:
        rgba = rgba[:, ::-1]                                           # chairs face the room centre
    dim, frac = PRIORS[cat]
    h0, w0 = rgba.shape[:2]
    s = (frac * (W if dim == "w" else H)) / (w0 if dim == "w" else h0)
    s *= 0.85 + 0.15 * (slot["y"] / 0.9)                               # farther (higher in image) = a bit smaller
    w, h = max(2, int(w0 * s)), max(2, int(h0 * s))
    if cat == "rug":
        h = int(h * 0.35)                                              # a rug on the floor, seen at an angle, looks squashed
    rgba = np.ascontiguousarray(cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA))
    x0 = int(slot["x"] * W - w / 2)
    if cat == "wall_art":
        y0, kind = int(slot["y"] * H - h / 2), "wall"
    elif cat == "rug":
        y0, kind = int(slot["y"] * H - h), "rug"
    else:
        y0, kind = int(slot["y"] * H - h), "floor"
    if kind != "rug":
        drop_shadow(canvas, rgba[..., 3], x0, y0 + h, kind)
    blend(canvas, rgba, x0, y0)


def render(lib, ids, slots, seed=0):
    """Compose the scene image from chosen item ids."""
    room = make_room(wall_tint([lib.summ[i] for i in ids]), seed)
    canvas = room.astype(np.float32).copy()
    # draw order: rug (on the floor) -> wall art -> everything else back-to-front by foot position
    order = sorted(range(len(slots)), key=lambda j: (slots[j]["cat"] != "rug", slots[j]["cat"] != "wall_art", slots[j]["y"]))
    for j in order:
        place(canvas, lib, ids[j], slots[j], room)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def compose(lib, prompt, harmony=True, K=5, seed=0):
    """Public entry: returns dict(image, ids, slots, scene, style, harmony, seconds)."""
    t0 = time.time()
    scene, style = parse_prompt(prompt)
    slots = SCENES[scene]
    ids, cands = choose_items(lib, slots, style, harmony, K)
    img = render(lib, ids, slots, seed)
    return dict(image=img, ids=ids, slots=slots, scene=scene, style=style, candidates=cands,
                harmony=harmony_score([lib.summ[i] for i in ids]), seconds=time.time() - t0)


# ---------------------------------------------------------------- 7. display helper
def item_card(lib, i, size=200):
    """Cut-out on a light background with its k-means palette swatches underneath (used by the app + figures)."""
    card = np.full((size + 28, size, 3), 244, np.uint8)
    c = lib.cutout(i)
    s = (size - 16) / max(c.shape[:2])
    c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y0, x0 = (size - c.shape[0]) // 2, (size - c.shape[1]) // 2
    a = c[..., 3:] / 255.0
    card[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] = (c[..., :3] * a + card[y0:y0 + c.shape[0], x0:x0 + c.shape[1]] * (1 - a)).astype(np.uint8)
    sw = lib.swatch(i)
    wd = size // max(len(sw), 1)
    for k, col in enumerate(sw):
        card[size + 4:size + 24, k * wd + 2:(k + 1) * wd - 2] = col
    return Image.fromarray(card)
