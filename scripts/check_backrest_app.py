"""One end-to-end callback smoke check; real products and one retrieval pass."""
import os,sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import app

built=app.run('cozy reading corner',True)
assert len(built)==17
state=built[6]
j=next(i for i,s in enumerate(state['slots']) if s['cat']=='chair')
def forbidden(*a,**k):
    raise AssertionError('Heavy model/retrieval/3D work was called during a 2D edit')
app.LIB.encode=forbidden
app.compose=forbidden
app.export_scene3d=forbidden
out=ROOT/'results/backrest'
out.mkdir(parents=True,exist_ok=True)
rows=[]
for angle in (90,135,180):
    t=time.perf_counter()
    r=app.apply_2d(state,False,0,3,120,j,angle,True)
    assert len(r)==8
    r[0].save(out/f'callback_scene_{angle}.png')
    r[5]['value'].save(out/f'callback_cutout_{angle}.png')
    rows.append(dict(angle=angle,ms=round((time.perf_counter()-t)*1000,2),message=r[4]))
reset=app.reset_2d(r[3])
assert np.array_equal(reset[0],built[0])
# Invalid part selections must be rejected rather than silently editing other products.
try:
    app.apply_2d(state,False,0,3,120,0,135,False)
except Exception as exc:
    assert 'chair' in str(exc).lower()
else:
    raise AssertionError('Non-chair edit was accepted')
(out/'callback_measurements.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows),flush=True)
print('PASS: real build -> 90/135/180 callback -> reset, source preserved, no model/3D calls while editing.',flush=True)
os._exit(0)
