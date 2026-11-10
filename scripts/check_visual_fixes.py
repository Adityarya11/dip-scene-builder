"""Focused checks for outward recline and desk-lamp scale/placement."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import backrest
from src.compose import Library, SCENES, render_scene, H

assert hasattr(backrest, 'resolve_direction'), 'Need chair-relative outward/inward direction'
assert backrest.resolve_direction('outward', True) == 'right'
assert backrest.resolve_direction('outward', False) == 'left'
assert backrest.resolve_direction('inward', True) == 'left'
lib = Library()
slots = [dict(s) for s in SCENES['home office']]
ids = [int(np.flatnonzero((lib.cat == s['cat']) & lib.ok)[0]) for s in slots]
ids[3], ids[4], ids[2] = 92, 302, 1253
img, report = render_scene(lib, ids, slots)
lamp, desk = report['items'][4], report['items'][2]
assert lamp['rgba'].shape[0] <= .20*H, 'Desk lamp still has floor-lamp height'
assert desk['bbox'][0] <= (lamp['bbox'][0]+lamp['bbox'][2])/2 <= desk['bbox'][2], 'Lamp not on desk'
assert desk['bbox'][1] <= lamp['bbox'][3] <= desk['bbox'][1]+.5*(desk['bbox'][3]-desk['bbox'][1]), 'Lamp base not on tabletop'
_, edited = render_scene(lib, ids, slots, edits={'backrest': {3: dict(angle=135, direction='outward')}})
assert edited['items'][3]['backrest']['direction'] == 'right'
assert np.array_equal(edited['items'][3]['M_net'], report['items'][3]['M_net'])
out=Path('results/backrest_sync'); out.mkdir(parents=True,exist_ok=True)
img.save(out/'office_fixed_upright.png')
print('PASS: outward direction, desk-lamp scale/support, fixed chair base.')
