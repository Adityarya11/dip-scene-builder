"""Focused check of the 2D geometric edits (spacing + rotation). Not a test suite: it saves presentation images
and prints MEASURED numbers.   python scripts/verify_2d.py
Output: results/presentation_2d/*.png and results/presentation_2d/measurements.json"""
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import cv2
import numpy as np
from PIL import Image, ImageDraw

import app
from src import compose as C
from src.edit2d import rotate_rgba

OUT = ROOT / "results/presentation_2d"
OUT.mkdir(parents=True, exist_ok=True)
LIB = app.LIB
M = {}                                              # everything printed is also saved here


def painted_mask(report, j):
    """Independent measurement: paste item j's opaque pixels (alpha>128) onto an empty canvas and read its extent."""
    it = report["items"][j]
    canvas = np.zeros((C.H, C.W), bool)
    rgba, x0, y0 = it["rgba"], it["x0"], it["y0"]
    h, w = rgba.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, C.W), min(y0 + h, C.H)
    canvas[Y0:Y1, X0:X1] = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0, 3] > 128
    ys, xs = np.where(canvas)
    return int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1, int(canvas.sum())


def measured_gap(report, a, b):
    xa0, xa1, *_ = painted_mask(report, a)
    xb0, xb1, *_ = painted_mask(report, b)
    return (xb0 - xa1) if report["gap"]["side"] == "right" else (xa0 - xb1)   # side chosen BEFORE the move


def titled(img, text):
    out = Image.new("RGB", (img.width, img.height + 26), (255, 255, 255))
    out.paste(img, (0, 26))
    ImageDraw.Draw(out).text((6, 6), text, fill=(0, 0, 0))
    return out


def hstack(imgs):
    w = sum(i.width for i in imgs) + 8 * (len(imgs) - 1)
    out = Image.new("RGB", (w, max(i.height for i in imgs)), (255, 255, 255))
    x = 0
    for i in imgs:
        out.paste(i, (x, 0)); x += i.width + 8
    return out


# ------------------------------------------------------------------ build one real scene (the heavy stage, once)
t = time.perf_counter()
res = app.run("cozy reading corner", True)
M["build_scene_seconds"] = round(time.perf_counter() - t, 2)
scene_img, state = res[0], res[6]
ids, slots, labels = state["ids_on"], state["slots"], state["labels"]
shelf, lamp, chair = [next(j for j, s in enumerate(slots) if s["cat"] == c) for c in ("shelf", "lamp", "chair")]
print("scene:", state["scene"], labels)

# ------------------------------------------------------------------ 0. default = unchanged image
img0, rep0 = C.render_scene(LIB, ids, slots, 0, {"rot": {}})
M["default_edit_identical_to_generated_image"] = bool(np.array_equal(np.array(img0), np.array(scene_img)))

# ------------------------------------------------------------------ 1. spacing shelf - lamp at two feasible values
lim = C.gap_limits(LIB, ids, slots, {}, shelf, lamp)
M["shelf_lamp_limits"] = {k: lim[k] for k in ("side", "lo", "hi", "current")}
spacing = []
frames = [titled(scene_img, f"generated layout (gap {lim['current']} px)")]
for want in (60, 240):
    img, rep = C.render_scene(LIB, ids, slots, 0, {"gap": dict(a=shelf, b=lamp, px=want)}, overlay=True)
    g = rep["gap"]
    mg = measured_gap(rep, shelf, lamp)
    spacing.append(dict(requested=want, achieved=g["achieved"], measured_from_pixels=mg, feasible=g["exact"]))
    frames.append(titled(img, f"requested {want} px, measured {mg} px"))
    img.save(OUT / f"spacing_shelf_lamp_{want}px.png")
# infeasible request: must report the feasible gap, not claim success
img, rep = C.render_scene(LIB, ids, slots, 0, {"gap": dict(a=shelf, b=lamp, px=700)}, overlay=True)
g = rep["gap"]
spacing.append(dict(requested=700, achieved=g["achieved"], measured_from_pixels=measured_gap(rep, shelf, lamp), feasible=g["exact"]))
img.save(OUT / "spacing_shelf_lamp_700px_infeasible.png")
M["spacing_shelf_lamp"] = spacing
M["spacing_unrelated_objects_unchanged"] = all(
    rep["items"][j]["bbox"] == rep0["items"][j]["bbox"] for j in range(len(slots)) if j != lamp)
hstack(frames).save(OUT / "spacing_before_after.png")

# ------------------------------------------------------------------ 2. chair rotation 0 / 90 / 135 / 180 (+45)
src_hash = hashlib.md5(LIB.cutout(ids[chair]).tobytes()).hexdigest()
rows, frames = [], []
for ang in (0, 45, 90, 135, 180):
    img, rep = C.render_scene(LIB, ids, slots, 0, {"rot": {chair: ang}}, overlay=True)
    it = rep["items"][chair]
    x0, y0, x1, y1 = it["bbox"]
    a = it["rgba"][..., 3]
    rows.append(dict(angle=ang, opaque_bbox=[x0, y0, x1, y1], inside_canvas=bool(x0 >= 0 and y0 >= 0 and x1 <= C.W and y1 <= C.H),
                     opaque_pixels=int((a > 128).sum()), corner_alpha=[int(a[0, 0]), int(a[0, -1]), int(a[-1, 0]), int(a[-1, -1])],
                     feet_y=int(y1), moved_to_fit=bool(it["clamped"])))
    crop = img.crop((max(0, x0 - 90), max(0, y0 - 60), min(C.W, x1 + 90), min(C.H, y1 + 40)))
    frames.append(titled(crop.resize((int(crop.width * 1.4), int(crop.height * 1.4)), Image.LANCZOS), f"chair {ang} deg"))
    img.save(OUT / f"chair_rot_{ang}.png")
base = rows[0]["opaque_pixels"]
for r in rows:
    r["opaque_area_vs_0deg"] = round(r["opaque_pixels"] / base, 3)
M["chair_rotation"] = rows
hstack([f for f, a in zip(frames, (0, 45, 90, 135, 180)) if a in (0, 90, 135, 180)]).save(OUT / "chair_rotation_before_after.png")
# source reuse: rotating there and back gives the original image exactly (no cumulative damage)
C.render_scene(LIB, ids, slots, 0, {"rot": {chair: 135}})
back, _ = C.render_scene(LIB, ids, slots, 0, {"rot": {chair: 0}})
M["source_cutout_unchanged_after_edits"] = hashlib.md5(LIB.cutout(ids[chair]).tobytes()).hexdigest() == src_hash
M["rotate_there_and_back_identical_to_original"] = bool(np.array_equal(np.array(back), np.array(scene_img)))

# ------------------------------------------------------------------ 3. halo check: premultiplied vs straight alpha
src = LIB.cutout(ids[chair])
pre, _ = rotate_rgba(src, 45)
h, w = src.shape[:2]
Mx = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 1.0)
nw, nh = int(np.ceil(h * abs(Mx[0, 1]) + w * abs(Mx[0, 0]))), int(np.ceil(h * abs(Mx[0, 0]) + w * abs(Mx[0, 1])))
Mx[0, 2] += nw / 2 - w / 2; Mx[1, 2] += nh / 2 - h / 2
straight = cv2.warpAffine(src, Mx, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
def edge_luma(im):
    a = im[..., 3]
    m = (a > 10) & (a < 120)                       # half-transparent edge pixels
    return float(im[..., :3][m].mean()) if m.any() else float("nan")
inner = float(src[..., :3][src[..., 3] > 250].mean())
M["halo_check_real_chair_45deg"] = dict(interior_mean=round(inner, 1), edge_mean_premultiplied=round(edge_luma(pre), 1),
                                        edge_mean_straight_alpha=round(edge_luma(straight), 1),
                                        note="real cut-outs store near-neutral colour under transparent pixels, so the two agree here")
# synthetic worst case: opaque red disc, transparent pixels carry WHITE colour (what a background removal can leave behind)
yy, xx = np.mgrid[0:120, 0:120]
disc = np.zeros((120, 120, 4), np.uint8)
disc[..., :3] = 255
inside = (yy - 60) ** 2 + (xx - 60) ** 2 < 40 ** 2
disc[inside, :3] = (200, 30, 30)
disc[inside, 3] = 255
pre_d, _ = rotate_rgba(disc, 45)
Md = cv2.getRotationMatrix2D((60, 60), 45, 1.0)
Md[0, 2] += (pre_d.shape[1] - 120) / 2; Md[1, 2] += (pre_d.shape[0] - 120) / 2
str_d = cv2.warpAffine(disc, Md, (pre_d.shape[1], pre_d.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255, 0))
def pink(im):                                       # distance of half-transparent edge pixels from the true red
    m = (im[..., 3] > 30) & (im[..., 3] < 225)
    return float(np.abs(im[..., :3][m].astype(float) - np.array([200, 30, 30])).mean()) if m.any() else float("nan")
M["halo_check_synthetic_red_disc"] = dict(edge_colour_error_premultiplied=round(pink(pre_d), 1), edge_colour_error_straight_alpha=round(pink(str_d), 1),
                                          note="mean abs RGB distance of edge pixels from the true colour; lower is better")

# ------------------------------------------------------------------ 4. all 8 classes: rotation + spacing on a synthetic 8-object layout
all_slots = [C.S("wall_art", "", .15, .30), C.S("shelf", "", .08, .80), C.S("sofa", "", .30, .80), C.S("table", "", .45, .95),
             C.S("chair", "", .62, .92), C.S("lamp", "", .78, .84), C.S("plant_vase", "", .92, .88), C.S("rug", "", .50, .92)]
all_ids = [int(np.where((LIB.cat == s["cat"]) & LIB.ok)[0][3]) for s in all_slots]
per_class = []
for j, s in enumerate(all_slots):
    for ang in (90, 135, 180):
        _, rep = C.render_scene(LIB, all_ids, all_slots, 0, {"rot": {j: ang}})
        it = rep["items"][j]
        x0, y0, x1, y1 = it["bbox"]
        _, rep0_ = C.render_scene(LIB, all_ids, all_slots, 0, {"rot": {}})
        base_px = int((rep0_["items"][j]["rgba"][..., 3] > 128).sum())
        per_class.append(dict(cls=s["cat"], angle=ang, inside=bool(x0 >= 0 and y0 >= 0 and x1 <= C.W and y1 <= C.H),
                              oversize_flagged=bool(it["oversize"]),
                              area_ratio=round(int((it["rgba"][..., 3] > 128).sum()) / base_px, 3),
                              transparent_corners=bool(it["rgba"][0, 0, 3] == 0 and it["rgba"][-1, -1, 3] == 0)))
M["rotation_all_classes"] = dict(n_checks=len(per_class), inside_canvas=sum(r["inside"] for r in per_class),
                                 outside_but_flagged_oversize=sum((not r["inside"]) and r["oversize_flagged"] for r in per_class),
                                 outside_and_not_flagged=sum((not r["inside"]) and not r["oversize_flagged"] for r in per_class),
                                 area_ratio_min=min(r["area_ratio"] for r in per_class), area_ratio_max=max(r["area_ratio"] for r in per_class),
                                 detail=per_class)
pairs = []
for a in range(len(all_slots)):
    for b in range(len(all_slots)):
        if a == b:
            continue
        lim_ = C.gap_limits(LIB, all_ids, all_slots, {}, a, b)
        want = min(40, max(lim_["hi"], 0))
        _, rep = C.render_scene(LIB, all_ids, all_slots, 0, {"gap": dict(a=a, b=b, px=want)})
        pairs.append(dict(a=all_slots[a]["cat"], b=all_slots[b]["cat"], requested=want, achieved=rep["gap"]["achieved"],
                          measured=measured_gap(rep, a, b), ok=measured_gap(rep, a, b) == rep["gap"]["achieved"]))
M["spacing_all_class_pairs"] = dict(n_pairs=len(pairs), measured_equals_reported=sum(p["ok"] for p in pairs),
                                    infeasible_pairs=[p for p in pairs if not p["ok"]][:5])
_, rep_all = C.render_scene(LIB, all_ids, all_slots, 0, {"rot": {4: 90}, "gap": dict(a=1, b=5, px=100)}, overlay=True)
C.render_scene(LIB, all_ids, all_slots, 0, {"rot": {j: 90 for j in range(8)}})  # smoke: rotate everything at once
# ------------------------------------------------------------------ 5. editing does NOT run retrieval / models
calls = []
def boom(name):
    def f(*a, **k):
        calls.append(name); raise RuntimeError(name)
    return f
saved = (LIB.encode, app.compose, app.export_scene3d, app.run)
LIB.encode, app.compose, app.export_scene3d = boom("clip_text"), boom("compose"), boom("export_scene3d")
import src.features as F, src.segment as S
saved_f = (F.clip_text, F.clip_images, S.rembg_mask)
F.clip_text, F.clip_images, S.rembg_mask = boom("clip_text"), boom("clip_images"), boom("rembg")
ok = True
try:
    app.apply_2d(state, True, shelf, lamp, 150, chair, 90, True)
except RuntimeError as e:
    ok = False
LIB.encode, app.compose, app.export_scene3d = saved[:3]
F.clip_text, F.clip_images, S.rembg_mask = saved_f
M["edit_callback_ran_without_retrieval_or_models"] = bool(ok and not calls)

# ------------------------------------------------------------------ 6. latency of the edit callback
def timed(fn, n):
    v = []
    for _ in range(n):
        t = time.perf_counter(); fn(); v.append((time.perf_counter() - t) * 1000)
    return v
C._LAYERS.clear(); C._ROOMS.clear()
cold = timed(lambda: app.apply_2d(state, True, shelf, lamp, 150, chair, 90, True), 1)[0]
angles = [0, 45, 90, 135, 180]
k = [0]
def varied():
    k[0] += 1
    app.apply_2d(state, True, shelf, lamp, 60 + (k[0] * 37) % 300, chair, angles[k[0] % 5], bool(k[0] % 2))
varied()
warm = timed(varied, 30)
rs = timed(lambda: C.render_scene(LIB, ids, slots, 0, {"rot": {chair: 90}, "gap": dict(a=shelf, b=lamp, px=150)}), 30)
M["latency_ms"] = dict(first_edit_cold_cache=round(cold, 1), apply_2d_median=round(statistics.median(warm), 1),
                       apply_2d_max=round(max(warm), 1), render_scene_only_median=round(statistics.median(rs), 1),
                       note="apply_2d renders BOTH harmony images, builds the PNG and the matrices; n=30 varied edits, CPU laptop")

# ------------------------------------------------------------------ 7. combined before/after image for the slides
both = C.render_scene(LIB, ids, slots, 0, {"rot": {chair: 90}, "gap": dict(a=shelf, b=lamp, px=440)}, overlay=True)[0]
hstack([titled(scene_img, "before: generated 2D scene"), titled(both, "after: chair rotated 90 deg, shelf/lamp gap 440 px")]).save(OUT / "before_after_combined.png")
C.compare_strip(LIB, ids, slots, chair, 135).save(OUT / "cutout_original_vs_rotated_135.png")
json.dump(M, open(OUT / "measurements.json", "w"), indent=2, default=str)
print(json.dumps({k: v for k, v in M.items() if k not in ("rotation_all_classes", "spacing_all_class_pairs")}, indent=1, default=str))
print("all classes rotation:", {k: v for k, v in M["rotation_all_classes"].items() if k != "detail"})
print("all class pairs spacing:", M["spacing_all_class_pairs"])
