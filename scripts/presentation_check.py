"""Generate a small real-output presentation pack, not a full test suite."""
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import sys
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw
import numpy as np
import app

out = ROOT / 'results/presentation'
out.mkdir(parents=True, exist_ok=True)
rows = []
for j, prompt in enumerate(['cozy reading corner', 'modern living room', 'minimalist home office']):
    result = app.run(prompt, True)
    selected, cards, message, glb, off, on = result[:6]   # run() also returns the 2D-edit state/controls
    selected.save(out / f'scene_{j+1}.png')
    pair = Image.new('RGB', (1536, 552), 'white')
    pair.paste(off, (0, 40)); pair.paste(on, (768, 40))
    draw = ImageDraw.Draw(pair)
    draw.text((16, 12), f'{prompt} | Harmony OFF', fill='black')
    draw.text((784, 12), 'Harmony ON', fill='black')
    pair.save(out / f'comparison_{j+1}.png')
    (out / f'room_{j+1}_on.glb').write_bytes(Path(glb).read_bytes())
    rows.append(dict(prompt=prompt, message=message,
                     pixel_changed=bool(np.any(np.asarray(off) != np.asarray(on)))))
    (out / 'harmony_comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows[-1]), flush=True)
# Exercise OFF route and check its GLB differs, not only the image.
r = app.run('cozy reading corner', False)
(out / 'room_1_off.glb').write_bytes(Path(r[3]).read_bytes())
assert (out/'room_1_off.glb').read_bytes() != (out/'room_1_on.glb').read_bytes()
assert all(row['pixel_changed'] for row in rows)
print('VERIFIED: real app ON/OFF routes, 3 scene comparisons, distinct GLB exports.', flush=True)
# Some native ML libraries on this Windows host hang in interpreter shutdown.
os._exit(0)
