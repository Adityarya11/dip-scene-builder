"""ONE command to show the complete pipeline live, stage by stage, with timings.

    python run_demo.py                          # default prompt, then opens the Gradio app
    python run_demo.py "modern living room"     # another prompt
    python run_demo.py --no-app                 # stop after the printed pipeline (no browser)
    python run_demo.py --rebuild                # ALSO re-run the full offline build first (src.data/build/palette/eval, slow)

Stages (each prints what it does and how long it took; pictures go to results/demo/):
  1  DIP enhancement      resize, median+Gaussian, gray-world, CLAHE          (live on sample photos)
  2  DIP segmentation     Otsu + morphology (+GrabCut) vs U2-Net, IoU          (live on the same photos)
  3  Descriptors          HSV histogram, HOG, LBP, k-means palette in LAB
  4  Retrieval + harmony  CLIP text -> top-K per slot -> best colour-harmonious set
  5  2D composition       placement rules, Reinhard colour match, shadows, alpha blending
  6  2D geometric edits   backrest-only deformation + horizontal spacing (measured)
  7  3D export            proxy room as GLB
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

try:                                              # product titles contain odd characters; never crash a live demo on print
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "2")

ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?", default="cozy reading corner")
ap.add_argument("--no-app", action="store_true", help="do not launch the Gradio app at the end")
ap.add_argument("--rebuild", action="store_true", help="re-run the full offline build first (about 25 min on a T4, much longer on CPU)")
ap.add_argument("--samples", type=int, default=4, help="photos used for the live stage 1-3 demo")
args = ap.parse_args()

OUT = ROOT / "results/demo"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.perf_counter()


def stage(n, title):
    print(f"\n=== Stage {n}: {title} " + "=" * max(3, 60 - len(title)), flush=True)


def took(t):
    return f"{(time.perf_counter() - t) * 1000:.0f} ms"


if args.rebuild:
    stage(0, "full offline rebuild")
    for cmd in (["-m", "src.data", "--per-cat", "190"], ["-m", "src.build", "--data", "data", "--out", "artifacts"],
                ["-m", "src.palette", "--art", "artifacts"], ["-m", "src.eval", "--data", "data", "--art", "artifacts", "--light"]):
        print("$ python", *cmd, flush=True)
        subprocess.run([sys.executable, *cmd], check=True)

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src import compose as C
from src.edit2d import ANGLES
from src.features import HANDCRAFTED, load_clip
from src.palette import dominant_colors, harmony_score
from src.preprocess import enhance, load_rgb, stages as pre_stages
from src.scene3d import export_scene3d
from src.segment import classical_mask, cutout, iou

ART = Path(os.environ.get("ART_DIR", "artifacts"))
meta = pd.read_csv(ART / "meta.csv")
print(f"Project: {len(meta)} product records, 8 categories. Prompt: '{args.prompt}'", flush=True)

# ------------------------------------------------------------------ stages 1-3: live DIP on a few real photos
data_dir = ROOT / "data"
if (data_dir / "meta.csv").exists():
    dm = pd.read_csv(data_dir / "meta.csv")
    sample = dm[dm.category.isin(["chair", "lamp", "sofa", "shelf", "rug", "table"])].groupby("category").head(1).head(args.samples)
    try:
        from src.segment import rembg_mask
        use_rembg = True
    except Exception:
        use_rembg = False
    rows = []
    stage(1, "DIP enhancement (resize -> median+Gaussian -> gray-world -> CLAHE)")
    shots = []
    for r in sample.itertuples():
        raw = load_rgb(data_dir / r.path_local)
        t = time.perf_counter()
        st = pre_stages(raw)
        print(f"  {r.category:<11} raw {raw.shape[1]}x{raw.shape[0]} -> enhanced {st['CLAHE'].shape[1]}x{st['CLAHE'].shape[0]}  ({took(t)})", flush=True)
        shots.append((r, raw, st["CLAHE"]))
    stage(2, "DIP segmentation (Otsu + morphology, GrabCut fallback) vs U2-Net")
    strip = []
    for r, raw, enh in shots:
        t = time.perf_counter()
        cm, method = classical_mask(enh)
        tc = took(t)
        msg = f"  {r.category:<11} classical[{method}] {tc}"
        rm = None
        if use_rembg:
            t = time.perf_counter()
            try:
                rm = rembg_mask(enh)
                msg += f" | U2-Net {took(t)} | IoU {iou(cm, rm):.2f} (U2-Net = reference, not ground truth)"
            except Exception as e:
                msg += f" | U2-Net unavailable ({type(e).__name__})"
        print(msg, flush=True)
        cut = cutout(enh, rm if rm is not None and (rm > 0).mean() > 0.03 else cm)
        bg = np.full((*cut.shape[:2], 3), 235, np.uint8)
        a = cut[..., 3:] / 255.0
        comp = (cut[..., :3] * a + bg * (1 - a)).astype(np.uint8)
        tile = [Image.fromarray(x).resize((170, 170)) for x in (raw, enh, np.dstack([cm] * 3), comp)]
        strip.append(tile)
        shots_feat = (enh, cm)
    sheet = Image.new("RGB", (170 * 4 + 30, (170 + 6) * len(strip)), "white")
    for i, tile in enumerate(strip):
        for k, im in enumerate(tile):
            sheet.paste(im, (k * 180, i * 176))
    sheet.save(OUT / "stages_raw_enhanced_mask_cutout.png")
    print(f"  saved {OUT / 'stages_raw_enhanced_mask_cutout.png'} (columns: raw | enhanced | classical mask | cut-out)", flush=True)
    stage(3, "Descriptors and palette")
    enh, cm = shots[0][2], classical_mask(shots[0][2])[0]
    for name, fn in HANDCRAFTED.items():
        t = time.perf_counter()
        v = fn(enh, cm) if name == "hsv" else fn(enh)
        print(f"  {name.upper():<4} descriptor: {len(v)} numbers ({took(t)})", flush=True)
    c = cutout(enh, cm)
    t = time.perf_counter()
    lab, w = dominant_colors(c)
    print(f"  k-means palette in LAB (k chosen by silhouette): {int((w > 0).sum())} colours ({took(t)})", flush=True)
else:
    print("\n(data/ not found: skipping the live photo stages 1-3; the stored cut-outs and vectors are used from stage 4.)", flush=True)

# ------------------------------------------------------------------ stage 4: retrieval + harmony
stage(4, "CLIP retrieval + colour-harmony search")
t = time.perf_counter()
clip = load_clip("cpu")
lib = C.Library(ART, clip=clip)
print(f"  loaded {len(lib.meta)} products + CLIP text model ({took(t)})", flush=True)
t = time.perf_counter()
scene, style = C.parse_prompt(args.prompt)
slots = [dict(s) for s in C.SCENES[scene]]                    # a COPY: the shared templates are never edited
print(f"  prompt -> template '{scene}', style words '{style or '-'}', slots: {', '.join(s['q'] for s in slots)}", flush=True)
ids_on, cands = C.choose_items(lib, slots, style, True)
ids_off = [int(c[0]) for c in cands]
print(f"  top-K per slot, best of {int(np.prod([len(c) for c in cands]))} sets ({took(t)})", flush=True)
labels = C.slot_labels(slots)
for lab_, j_on, j_off in zip(labels, ids_on, ids_off):
    print(f"   {lab_:<11} harmony ON  {lib.meta.image_id[j_on]:<13} {str(lib.meta.caption[j_on])[:46]}", flush=True)
print(f"  colour harmony score: off {harmony_score([lib.summ[i] for i in ids_off]):.3f}  ->  on {harmony_score([lib.summ[i] for i in ids_on]):.3f}", flush=True)

# ------------------------------------------------------------------ stage 5: 2D composition
stage(5, "2D composition (placement, Reinhard colour match, Gaussian shadows, alpha blending)")
t = time.perf_counter()
img_on, rep = C.render_scene(lib, ids_on, slots)
img_off, _ = C.render_scene(lib, ids_off, slots)
print(f"  rendered harmony-ON and harmony-OFF scenes ({took(t)})", flush=True)
img_on.save(OUT / "scene_harmony_on.png")
img_off.save(OUT / "scene_harmony_off.png")

# ------------------------------------------------------------------ stage 6: 2D geometric edits
stage(6, "2D geometric edits (guided backrest recline + spacing), no model is called")
pair = C.default_pair(slots)
rot_j = next((j for j, s in enumerate(slots) if s["cat"] == "chair"), None)
backrest = dict(angle=135, hinge=55, left=0, right=100, direction='left')
edits = {'backrest': {rot_j: backrest}} if rot_j is not None else {}
if pair:
    lim = C.gap_limits(lib, ids_on, slots, {}, *pair)
    want = min(lim["hi"], lim["current"] + 120) if lim["hi"] > 0 else 0
    edits["gap"] = dict(a=pair[0], b=pair[1], px=want)
t = time.perf_counter()
img_edit, rep2 = C.render_scene(lib, ids_on, slots, 0, edits, overlay=True)
print(f"  edit applied in {took(t)} (layers cached from stage 5)", flush=True)
if rot_j is not None:
    print(f"  backrest: {labels[rot_j]} 135 deg (2D reference), seat/legs fixed; matrix:\n"
          f"{np.array2string(rep2['items'][rot_j]['backrest']['M_backrest'], precision=3, suppress_small=True)}", flush=True)
else:
    print('  This layout has no chair; backrest edit skipped.', flush=True)
if pair and rep2["gap"]:
    g = rep2["gap"]
    print(f"  spacing: {labels[g['a']]} fixed, {labels[g['b']]} moved {g['side']}: requested {g['requested']} px, achieved {g['achieved']} px "
          f"(feasible 0..{g['hi']} px)", flush=True)
img_edit.save(OUT / "scene_2d_edited_overlay.png")
if rot_j is not None:
    C.compare_strip(lib, ids_on, slots, rot_j, backrest=backrest).save(OUT / 'cutout_backrest_135.png')

# ------------------------------------------------------------------ stage 7: 3D
stage(7, "3D export (proxy room, GLB)")
t = time.perf_counter()
glb = export_scene3d(lib, dict(ids=ids_on, slots=slots))
import shutil
shutil.copy(glb, OUT / "room.glb")
print(f"  exported {os.path.getsize(glb) // 1024} KB GLB ({took(t)}) -> {OUT / 'room.glb'}", flush=True)
print("  (the 3D room is made of simple shapes coloured from the products; it is not a reconstruction of the photos)", flush=True)

# ------------------------------------------------------------------ summary picture
sheet = Image.new("RGB", (768 * 3 + 20, 512 + 30), "white")
for k, (im, txt) in enumerate(((img_off, "harmony OFF"), (img_on, "harmony ON"), (img_edit, "2D edits: backrest 135 deg + spacing"))):
    sheet.paste(im, (k * 778, 30))
    ImageDraw.Draw(sheet).text((k * 778 + 6, 8), txt, fill=(0, 0, 0))
sheet.save(OUT / "demo_summary.png")
print(f"\nDone in {time.perf_counter() - T0:.1f} s. Pictures and the GLB are in {OUT}", flush=True)

if not args.no_app:
    print("\nLaunching the Gradio app (Ctrl+C to stop)...", flush=True)
    import app
    app.demo.queue(default_concurrency_limit=1).launch(inbrowser=True)
