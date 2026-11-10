"""Step 4b - prompt -> scene image (the "2.5D" composer).

  prompt --keywords--> slots --CLIP text search--> top-K per slot --brute-force best combo-->
  placement rules --> procedural room --> colour match (Reinhard) --> drop shadow --> alpha blend

Everything after retrieval is plain NumPy/OpenCV image processing.
"""
import re
import time

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.edit2d import affine_chain, opaque_bbox, rotate_rgba, solve_gap, to3
from src.backrest import recline_rgba
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
        if "image_id" in p:
            lookup = {str(iid): j for j, iid in enumerate(p["image_id"])}
            rows = [lookup[str(iid)] for iid in self.meta.image_id]
        elif len(p["lab"]) == len(self.meta):
            rows = np.arange(len(self.meta))
        else:
            raise ValueError("Stale/misaligned palettes: run python scripts/refresh_palettes.py before deployment.")
        self.pal_lab, self.pal_w = p["lab"][rows], p["w"][rows]
        self.summ = [item_summary(l, w) for l, w in zip(self.pal_lab, self.pal_w)]
        self.index = Index(self.emb, "clip")
        self.cat = self.meta.category.values
        # quality filter: if the two segmenters disagree (IoU < 0.3) the cut-out is probably bad -> never use it in a scene
        # (wall art uses a rectangle mask instead, so it is exempt)
        iou = self.meta.get("iou_classical_vs_rembg", pd.Series(1.0, index=self.meta.index)).values
        quality_ok = (iou >= 0.3) | (self.cat == "wall_art")

        cutout_exists = np.array([
            (self.art / "cutouts" / f"{image_id}.webp").is_file()
            for image_id in self.meta.image_id
        ])
        real_product = ~self.meta.caption.fillna("").str.contains("swatch", case=False).to_numpy()
        self.ok = quality_ok & cutout_exists & real_product & (self.pal_w.sum(axis=1) > 0)
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


def choose_items(lib, slots, style, harmony=True, K=8):
    """Top-K candidates per slot by CLIP text similarity (restricted to the slot's category),
    then brute-force every combination (K^slots <= ~15k) and keep the best.
        combo score = mean_i s_i + LAMBDA * harmony(combo)
    where s_i = (sim_i - best_sim_of_slot) / 0.05 so that CLIP differences and harmony are on similar scales.
    With harmony=False the harmony term is dropped -> simply the top-1 item of every slot."""
    qv = lib.encode([f"a photo of a {style} {s['q']}".replace("  ", " ") for s in slots])
    cands, sims = [], []
    for s, q in zip(slots, qv):
        ids, sc = lib.index.search(q, K, mask=(lib.cat == s["cat"]) & lib.ok)
        keep = np.isfinite(sc)
        ids, sc = ids[keep], sc[keep]
        if not len(ids):
            raise ValueError(f"No usable products for {s['cat']}; check cutouts and palettes.")
        keep = sc >= sc[0] - 0.04  # never trade away semantic relevance for colour
        ids, sc = ids[keep], sc[keep]
        cands.append(ids)
        sims.append((sc - sc[0]) / 0.05)
    if not harmony:
        return [int(c[0]) for c in cands], cands
    # pair scores of all candidate items, computed once
    allids = sorted({int(i) for c in cands for i in c})
    pos = {i: j for j, i in enumerate(allids)}
    P = np.array([[pair_score(lib.summ[a], lib.summ[b]) for b in allids] for a in allids])
    # Exact same objective, evaluated in NumPy rather than slow Python loops.
    grid = np.indices(tuple(len(c) for c in cands)).reshape(len(cands), -1)
    clip_s = sum(sims[j][grid[j]] for j in range(len(cands))) / len(cands)
    pp = [np.array([pos[int(i)] for i in c])[grid[j]] for j, c in enumerate(cands)]
    pairs = [(a, b) for a in range(len(pp)) for b in range(a + 1, len(pp))]
    hs = sum(P[pp[a], pp[b]] for a, b in pairs) / max(1, len(pairs))
    best = int(np.argmax(clip_s + LAMBDA * hs))
    return [int(c[grid[j, best]]) for j, c in enumerate(cands)], cands


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
    ab = [(np.cos(np.radians(s[0])) * s[1], np.sin(np.radians(s[0])) * s[1]) for s in items_summ if not s[2]]
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


# Small caches so that 2D edits (spacing / rotation) re-use the room and the colour-matched layers.
_ROOMS, _LAYERS = {}, {}


def _remember(cache, key, make, limit):
    """Tiny FIFO cache: keeps memory small on the free Space."""
    if key not in cache:
        if len(cache) >= limit:
            cache.pop(next(iter(cache)))
        cache[key] = make()
    return cache[key]


def get_room(items_summ, seed=0):
    tint = wall_tint(items_summ)
    key = (float(tint[0]), float(tint[1]), seed)
    return _remember(_ROOMS, key, lambda: make_room(tint, seed), 6), key


def make_layer(lib, idx, slot, room, room_key, rot=0):
    """Colour-matched, scaled (and optionally rotated) RGBA layer of ONE product.
    Always starts from the cached SOURCE cut-out, never from an earlier transformed layer, so repeated
    edits cannot accumulate interpolation damage. rot is a 2D image-plane rotation in degrees."""
    cat = slot["cat"]
    flip = cat == "chair" and slot["x"] > 0.6                            # chairs face the room centre
    key = (id(lib), int(idx), cat, flip, round(float(slot["y"]), 4), int(rot), room_key)

    def build():
        rgba = reinhard(lib.cutout(idx), room)
        if flip:
            rgba = rgba[:, ::-1]
        dim, frac = PRIORS[cat]
        h0, w0 = rgba.shape[:2]
        s = (frac * (W if dim == "w" else H)) / (w0 if dim == "w" else h0)
        s *= 0.85 + 0.15 * (slot["y"] / 0.9)                             # farther (higher in image) = a bit smaller
        w, h = max(2, int(w0 * s)), max(2, int(h0 * s))
        if cat == "rug":
            h = int(h * 0.35)                                            # a rug on the floor, seen at an angle, looks squashed
        rgba = np.ascontiguousarray(cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA))
        M_rot = None
        if int(rot) % 360:
            rgba, M_rot = rotate_rgba(rgba, rot)                         # affine rotation, premultiplied alpha
        return dict(rgba=rgba, bbox=opaque_bbox(rgba[..., 3]), flip=flip, src_size=(w0, h0), scaled=(w, h),
                    M_rot=M_rot, rot=int(rot))
    return _remember(_LAYERS, key, build, 64)


def place_item(lib, idx, slot, room, room_key, rot=0):
    """Where does ONE layer go? Unrotated items use the original rule (so default output is unchanged);
    rotated items are grounded by their opaque bounding box and kept inside the canvas."""
    L = make_layer(lib, idx, slot, room, room_key, rot)
    rgba, cat = L["rgba"], slot["cat"]
    h, w = rgba.shape[:2]
    bx0, by0, bx1, by1 = L["bbox"]
    clamped = oversize = False
    if int(rot) % 360 == 0:
        x0 = int(slot["x"] * W - w / 2)
        y0 = int(slot["y"] * H - h / 2) if cat == "wall_art" else int(slot["y"] * H - h)
        base = y0 + h
    else:
        x0 = int(round(slot["x"] * W - (bx0 + bx1) / 2))
        y0 = (int(round(slot["y"] * H - (by0 + by1) / 2)) if cat == "wall_art" else int(slot["y"] * H) - by1)
        cx, cy = _clamp(x0, -bx0, W - bx1), _clamp(y0, -by0, H - by1)  # keep the opaque box on the canvas
        clamped = (cx, cy) != (x0, y0)
        oversize = (W - bx1) < -bx0 or (H - by1) < -by0                # rotated box larger than the canvas itself
        x0, y0 = cx, cy
        base = y0 + h if cat == "wall_art" else y0 + by1
    kind = "wall" if cat == "wall_art" else ("rug" if cat == "rug" else "floor")
    M = affine_chain(L["src_size"], L["flip"], L["scaled"], L["M_rot"], (x0, y0))
    return dict(rgba=rgba, x0=x0, y0=y0, base=base, kind=kind, cat=cat, bbox_l=L["bbox"], rot=int(rot),
                M_net=M, M_rot=L["M_rot"], clamped=clamped, oversize=oversize, idx=int(idx))


def _clamp(v, lo, hi):
    """Clamp v to [lo, hi]. lo > hi means the object is bigger than the canvas on this axis: centre it."""
    return (lo + hi) // 2 if lo > hi else min(max(v, lo), hi)


def place_backrest_item(lib, idx, slot, room, room_key, config):
    """Articulate only the backrest; preserve the original seat/legs canvas coordinates."""
    it = place_item(lib, idx, slot, room, room_key, 0)
    if slot['cat'] != 'chair':
        raise ValueError('Backrest editing is available for chair slots only.')
    cfg = {k: config.get(k, v) for k, v in dict(angle=90, hinge=55, left=0, right=100, direction='left').items()}
    key = ('backrest', id(lib), int(idx), tuple(sorted(cfg.items())), slot['x'], slot['y'], room_key)
    rgba, info = _remember(_LAYERS, key, lambda: recline_rgba(it['rgba'], **cfg), 64)
    original_x, original_y = it['x0'], it['y0']
    ox, oy = info['offset']
    it.update(rgba=rgba, x0=original_x+ox, y0=original_y+oy, bbox_l=opaque_bbox(rgba[..., 3]), backrest=info)
    source_local = it['M_net'].copy()
    source_local[:, 2] -= [original_x, original_y]
    T = np.array([[1., 0., it['x0']], [0., 1., it['y0']]])
    it['M_backrest_net'] = (to3(T) @ to3(info['M_backrest']) @ to3(source_local))[:2]
    box = bbox_canvas(it)
    it['clipped'] = box[0] < 0 or box[1] < 0 or box[2] > W or box[3] > H
    return it


def bbox_canvas(it):
    b = it["bbox_l"]
    return it["x0"] + b[0], it["y0"] + b[1], it["x0"] + b[2], it["y0"] + b[3]


CAT_NAMES = {"sofa": "Sofa", "chair": "Chair", "table": "Table", "lamp": "Lamp", "rug": "Rug",
             "plant_vase": "Plant/vase", "shelf": "Shelf", "wall_art": "Wall art"}


def slot_labels(slots):
    """'Chair 1', 'Chair 2' for repeated categories, plain 'Lamp' when the category occurs once."""
    total = {}
    for s in slots:
        total[s["cat"]] = total.get(s["cat"], 0) + 1
    seen, out = {}, []
    for s in slots:
        seen[s["cat"]] = seen.get(s["cat"], 0) + 1
        out.append(CAT_NAMES[s["cat"]] + (f" {seen[s['cat']]}" if total[s["cat"]] > 1 else ""))
    return out


def default_pair(slots):
    """Shelf + lamp when both exist ("rack" = shelf category); else the first two floor objects; else None."""
    cats = [s["cat"] for s in slots]
    if "shelf" in cats and "lamp" in cats:
        return cats.index("shelf"), cats.index("lamp")
    other = [j for j, c in enumerate(cats) if c not in ("rug", "wall_art")] or list(range(len(cats)))
    return (other[0], other[1]) if len(other) >= 2 else None


def render_scene(lib, ids, slots, seed=0, edits=None, overlay=False):
    """Compose the scene. edits = {"rot": {slot_index: degrees}, "gap": {"a": i, "b": j, "px": int or None}}.
    Returns (PIL image, report). With edits=None the output equals the original renderer."""
    edits = edits or {}
    rots = edits.get("rot", {})
    backrests = edits.get('backrest', {})
    room, room_key = get_room([lib.summ[i] for i in ids], seed)
    canvas = room.astype(np.float32).copy()
    items = [place_backrest_item(lib, i, s, room, room_key, backrests[j]) if j in backrests
             else place_item(lib, i, s, room, room_key, int(rots.get(j, 0)))
             for j, (i, s) in enumerate(zip(ids, slots))]

    gap = None
    g = edits.get("gap")
    if g and g.get("a") is not None and g.get("b") is not None and g["a"] != g["b"] and max(g["a"], g["b"]) < len(items):
        A, B = items[g["a"]], items[g["b"]]
        ba, bb = bbox_canvas(A), bbox_canvas(B)
        cur = (bb[0] - ba[2]) if (bb[0] + bb[2]) >= (ba[0] + ba[2]) else (ba[0] - bb[2])
        sol = solve_gap(ba, bb, cur if g.get("px") is None else g["px"], W)
        if g.get('px') is None:
            # A disabled spacing control must not move overlapping original items.
            sol.update(dx=0, achieved=int(cur), exact=True)
        B["x0"] += sol["dx"]                                      # translate B horizontally, nothing else moves
        B["M_net"] = B["M_net"].copy()
        B["M_net"][0, 2] += sol["dx"]
        if 'M_backrest_net' in B:
            B['M_backrest_net'] = B['M_backrest_net'].copy()
            B['M_backrest_net'][0, 2] += sol['dx']
        gap = dict(sol, a=g["a"], b=g["b"], applied=g.get("px") is not None)

    # draw order: rug (on the floor) -> wall art -> everything else back-to-front by foot position
    order = sorted(range(len(slots)), key=lambda j: (slots[j]["cat"] != "rug", slots[j]["cat"] != "wall_art", slots[j]["y"]))
    for j in order:
        it = items[j]
        if it["kind"] != "rug":
            drop_shadow(canvas, it["rgba"][..., 3], it["x0"], it["base"], it["kind"])   # shadow from the (transformed) alpha
        blend(canvas, it["rgba"], it["x0"], it["y0"])
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    labels = slot_labels(slots)
    report = dict(labels=labels, gap=gap,
                  items=[dict(label=labels[j], cat=it["cat"], rot=it["rot"], bbox=bbox_canvas(it), M_net=it["M_net"],
                              M_rot=it["M_rot"], clamped=it["clamped"], oversize=it["oversize"], x0=it["x0"], y0=it["y0"],
                              rgba=it["rgba"], backrest=it.get('backrest'),
                              M_backrest_net=it.get('M_backrest_net'), clipped=it.get('clipped', False))
                         for j, it in enumerate(items)])
    if overlay:
        img = draw_overlay(img, report)
    return img, report


def gap_limits(lib, ids, slots, rots, a, b, seed=0, backrests=None):
    """Feasible horizontal gap range for the pair (a fixed, b moves) at the current rotations.
    Needs only the two cached layers, no painting."""
    room, key = get_room([lib.summ[i] for i in ids], seed)
    A = place_item(lib, ids[a], slots[a], room, key, int(rots.get(a, 0)))
    B = place_item(lib, ids[b], slots[b], room, key, int(rots.get(b, 0)))
    backrests = backrests or {}
    if a in backrests:
        A = place_backrest_item(lib, ids[a], slots[a], room, key, backrests[a])
    if b in backrests:
        B = place_backrest_item(lib, ids[b], slots[b], room, key, backrests[b])
    ba, bb = bbox_canvas(A), bbox_canvas(B)
    cur = (bb[0] - ba[2]) if (bb[0] + bb[2]) >= (ba[0] + ba[2]) else (ba[0] - bb[2])
    return solve_gap(ba, bb, cur, W)


def draw_overlay(img, report):
    """'Show transformations': opaque bounding boxes, rotation angles and the horizontal gap measurement."""
    from PIL import ImageDraw
    img = img.copy()
    d = ImageDraw.Draw(img)
    for it in report["items"]:
        x0, y0, x1, y1 = it["bbox"]
        col = (230, 60, 60) if it["rot"] else (30, 150, 200)
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=col, width=2)
        text = it["label"] + (f"  rot {it['rot']} deg" if it["rot"] else "")
        if it.get('backrest'):
            text = it['label'] + f"  backrest {it['backrest']['angle']:g} deg (2D)"
        d.rectangle([x0, max(0, y0 - 13), x0 + d.textlength(text) + 6, max(13, y0)], fill=col)
        d.text((x0 + 3, max(0, y0 - 13)), text, fill=(255, 255, 255))
    g = report["gap"]
    if g:
        A, B = report["items"][g["a"]]["bbox"], report["items"][g["b"]]["bbox"]
        xa, xb = (A[2], B[0]) if g["side"] == "right" else (B[2], A[0])
        lo, hi = max(A[1], B[1]), min(A[3], B[3])
        y = int((lo + hi) / 2) if hi > lo else int((B[1] + B[3]) / 2)
        d.line([xa, y, xb, y], fill=(250, 200, 0), width=3)
        for x in (xa, xb):
            d.line([x, y - 8, x, y + 8], fill=(250, 200, 0), width=3)
        txt = f"gap {g['achieved']} px" + ("" if g["exact"] else f" (asked {g['requested']})")
        d.rectangle([min(xa, xb), y - 22, min(xa, xb) + d.textlength(txt) + 8, y - 8], fill=(40, 40, 40))
        d.text((min(xa, xb) + 4, y - 21), txt, fill=(250, 200, 0))
    return img


def render(lib, ids, slots, seed=0):
    """Compose the scene image from chosen item ids (no 2D edits)."""
    return render_scene(lib, ids, slots, seed)[0]


def compare_strip(lib, ids, slots, j, rot=0, seed=0, backrest=None):
    """Original cut-out -> transformed cut-out on a checkerboard, with opaque bounding boxes."""
    from PIL import ImageDraw
    room, key = get_room([lib.summ[i] for i in ids], seed)
    size = 230
    out = Image.new("RGB", (size * 2 + 40, size + 34), (245, 245, 242))
    d = ImageDraw.Draw(out)
    yy, xx = np.mgrid[0:size, 0:size]
    check = np.where(((yy // 12 + xx // 12) % 2)[..., None] == 0, 232, 205).astype(np.float32).repeat(3, axis=2)
    for k, (r, text) in enumerate(((0, "original cut-out"), (int(rot), f"after 2D rotation {int(rot)} deg"))):
        L = make_layer(lib, ids[j], slots[j], room, key, r)
        region = None
        if backrest is not None:
            text = 'source / backrest region' if k == 0 else f"backrest {backrest['angle']:g} deg (2D)"
            rgba, detail = recline_rgba(L['rgba'], **backrest)
            if k == 0:
                region = detail['region']
            else:
                L = dict(L, rgba=rgba, bbox=opaque_bbox(rgba[..., 3]))
        tile = check.copy()
        rgba = L["rgba"]
        s = (size - 24) / max(rgba.shape[:2])
        sm = cv2.resize(rgba, (max(1, int(rgba.shape[1] * s)), max(1, int(rgba.shape[0] * s))), interpolation=cv2.INTER_AREA)
        oy, ox = (size - sm.shape[0]) // 2, (size - sm.shape[1]) // 2
        a = sm[..., 3:] / 255.0
        tile[oy:oy + sm.shape[0], ox:ox + sm.shape[1]] = sm[..., :3] * a + tile[oy:oy + sm.shape[0], ox:ox + sm.shape[1]] * (1 - a)
        x0 = k * (size + 40)
        out.paste(Image.fromarray(tile.astype(np.uint8)), (x0, 30))
        bx0, by0, bx1, by1 = [int(v * s) for v in L["bbox"]]
        d.rectangle([x0 + ox + bx0, 30 + oy + by0, x0 + ox + bx1 - 1, 30 + oy + by1 - 1], outline=(230, 60, 60), width=2)
        if region is not None:
            rx0, ry0, rx1, ry1 = [int(v*s) for v in region]
            d.rectangle([x0+ox+rx0, 30+oy+ry0, x0+ox+rx1, 30+oy+ry1], outline=(0, 170, 200), width=2)
            d.line([x0+ox+rx0, 30+oy+ry1, x0+ox+rx1, 30+oy+ry1], fill=(255, 170, 0), width=3)
        d.text((x0 + 4, 8), text, fill=(30, 30, 30))
    d.text((size + 12, 30 + size // 2 - 6), "->", fill=(30, 30, 30))
    return out


def compose(lib, prompt, harmony=True, K=8, seed=0):
    """Public entry: returns dict(image, ids, slots, scene, style, harmony, seconds)."""
    t0 = time.time()
    scene, style = parse_prompt(prompt)
    slots = SCENES[scene]
    ids, cands = choose_items(lib, slots, style, harmony, K)
    img = render(lib, ids, slots, seed)
    return dict(image=img, ids=ids, slots=slots, scene=scene, style=style, candidates=cands,
                harmony=harmony_score([lib.summ[i] for i in ids]), harmony_enabled=bool(harmony), seconds=time.time() - t0)


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
