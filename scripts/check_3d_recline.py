"""CPU-only GLB articulation check: only back/headrest move around the hinge."""
from pathlib import Path
import sys
import numpy as np
import trimesh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.compose import Library, SCENES
from src.scene3d import export_scene3d
lib=Library()
slots=[dict(s) for s in SCENES['home office']]
ids=[int(np.flatnonzero((lib.cat==s['cat']) & lib.ok)[0]) for s in slots]
out=Path('results/backrest_sync'); out.mkdir(parents=True,exist_ok=True)

def vertices(scene,node):
    matrix, geom = scene.graph[node]
    return trimesh.transform_points(scene.geometry[geom].vertices,matrix)

scenes=[]
for angle in (90,135,180):
    result=dict(ids=ids,slots=slots,scene='home office',harmony_enabled=True,
                backrest={3:dict(angle=angle,direction='outward')})
    p=export_scene3d(lib,result,out/f'office_{angle}.glb')
    scenes.append(trimesh.load(p,force='scene'))
a,b,c=scenes
assert not np.allclose(vertices(a,'chair_0_back'),vertices(b,'chair_0_back')), 'GLB ignores backrest angle'
assert not np.allclose(vertices(a,'chair_0_headrest'),vertices(b,'chair_0_headrest')), 'Headrest did not follow backrest'
assert np.allclose(vertices(a,'chair_0_seat'),vertices(b,'chair_0_seat')), 'Seat moved'
assert np.allclose(vertices(a,'chair_0_leg_-1_-1'),vertices(b,'chair_0_leg_-1_-1')), 'Leg moved'
assert vertices(b,'chair_0_headrest')[:,1].mean() < vertices(a,'chair_0_headrest')[:,1].mean(), 'Not reclining down'
assert vertices(b,'chair_0_headrest')[:,2].mean() > vertices(a,'chair_0_headrest')[:,2].mean(), 'Office chair leans inward'
assert a.metadata['chair_backrests'][0]['angle']==90
assert c.metadata['chair_backrests'][0]['angle']==180
print('PASS: 90/135/180 GLBs; head/backrest outward articulation; fixed seat/legs; pose metadata.')
