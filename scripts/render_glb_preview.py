"""Render an exported GLB room to a static PNG (matplotlib painter's algorithm, no GPU or browser needed).

    python scripts/render_glb_preview.py results/demo/room.glb room_preview.png

Shell (floor, walls) is drawn first, then every convex box/cylinder back to front with back-face culling.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection

SHELL = ("wood_floor", "back_wall", "left_wall", "baseboard_back", "baseboard_left")


def render(glb, png, elev=27, azim=38):
    scene = trimesh.load(glb)
    az, el = np.radians(azim), np.radians(elev)
    cam_dir = np.array([np.sin(az) * np.cos(el), -np.cos(az) * np.cos(el), np.sin(el)])   # direction to the camera
    up = np.cross(np.cross(cam_dir, [0, 0, 1]), cam_dir)
    up /= np.linalg.norm(up)
    right = np.cross(up, cam_dir)
    right /= np.linalg.norm(right)
    items = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(T)
        v = np.asarray(m.vertices)
        v = np.c_[v[:, 0], -v[:, 2], v[:, 1]]                          # glTF Y-up -> x, depth, height
        fc = np.asarray(m.visual.face_colors)[:, :3] / 255.0
        shell = any(node.startswith(s) for s in SHELL)
        order = 0 if node.startswith("wood_floor") else (1 if shell else (1.5 if "_rug" in node else 2))
        items.append((order, v, m.faces, fc, m.face_normals))
    light = np.array([-0.35, -0.55, 0.75]); light /= np.linalg.norm(light)
    polys, cols, keys = [], [], []
    for order, v, faces, fc, fn in items:
        for f, c, n in zip(faces, fc, fn):
            n2 = np.array([n[0], -n[2], n[1]])
            if n2 @ cam_dir < 0:
                continue
            P = v[f]
            polys.append(np.c_[P @ right, P @ up])
            cols.append(np.clip(c * (0.58 + 0.42 * max(0.0, float(n2 @ light))), 0, 1))
            keys.append((order, round(float(v.mean(axis=0) @ cam_dir), 3), float((P @ cam_dir).mean())))
    idx = sorted(range(len(polys)), key=lambda i: keys[i])
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.add_collection(PolyCollection([polys[i] for i in idx], facecolors=[cols[i] for i in idx],
                                     edgecolors=[cols[i] * 0.8 for i in idx], linewidths=0.15))
    allp = np.concatenate(polys)
    ax.set_xlim(allp[:, 0].min() - .2, allp[:, 0].max() + .2)
    ax.set_ylim(allp[:, 1].min() - .2, allp[:, 1].max() + .2)
    ax.set_aspect("equal"); ax.axis("off"); fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(png, facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
