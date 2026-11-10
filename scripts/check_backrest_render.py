"""Small real-output check for backrest-only edits; no model inference needed."""
from pathlib import Path
import sys, time, json
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.compose import Library, SCENES, render_scene, compare_strip

lib = Library(ROOT / 'artifacts')
slots = [dict(s) for s in SCENES['reading corner']]
ids = [int(np.flatnonzero((lib.cat == s['cat']) & lib.ok)[0]) for s in slots]
j = next(j for j, s in enumerate(slots) if s['cat'] == 'chair')
# Prefer a real recliner for the saved illustration, not a synthetic proxy.
chair_ids = [int(i) for i in np.flatnonzero((lib.cat == 'chair') & lib.ok)
             if 'recliner' in str(lib.meta.caption[i]).lower()]
if chair_ids:
    ids[j] = chair_ids[0]
base, report = render_scene(lib, ids, slots)
fixed = report['items'][j]
out = ROOT / 'results/backrest'
out.mkdir(parents=True, exist_ok=True)
measurements = []
images = []
for angle in (90, 135, 180):
    cfg = dict(angle=angle, hinge=55, left=0, right=100, direction='left')
    t = time.perf_counter()
    img, r = render_scene(lib, ids, slots, edits={'backrest': {j: cfg}})
    ms = (time.perf_counter()-t)*1000
    it = r['items'][j]
    assert np.array_equal(it['M_net'], fixed['M_net']), 'Fixed seat/legs matrix moved'
    ox, oy = it['backrest']['offset']
    h, w = fixed['rgba'].shape[:2]
    row = it['backrest']['hinge']+2
    assert np.array_equal(it['rgba'][row-oy:h-oy, -ox:w-ox], fixed['rgba'][row:h]), 'Lower chair pixels changed'
    assert np.array_equal(render_scene(lib, ids, slots)[0], base), 'Cached source changed'
    if angle == 90:
        assert np.array_equal(img, base), '90 reference differs from source'
    img.save(out / f'scene_{angle}.png')
    compare_strip(lib, ids, slots, j, backrest=cfg).save(out / f'cutout_{angle}.png')
    tile = Image.new('RGB', (img.width, img.height+28), 'white')
    tile.paste(img, (0, 28))
    ImageDraw.Draw(tile).text((12, 8), f'Backrest {angle} degrees (guided 2D)', fill='black')
    images.append(tile)
    measurements.append(dict(angle=angle, milliseconds=round(ms, 2), fixed_base_verified=True))
strip = Image.new('RGB', (sum(im.width for im in images), images[0].height), 'white')
for n, im in enumerate(images):
    strip.paste(im, (n*im.width, 0))
strip.save(out / 'backrest_comparison.png')
(out/'measurements.json').write_text(json.dumps(measurements, indent=2))
print('REAL PRODUCT', lib.meta.image_id[ids[j]], lib.meta.caption[ids[j]])
print(json.dumps(measurements))
print('PASS: real-chair base pixels and canvas matrix unchanged at 90/135/180; source cache intact.')
