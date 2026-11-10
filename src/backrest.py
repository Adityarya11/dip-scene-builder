"""Guided 2D chair articulation: move only a backrest region, keep the base fixed.

This is a piecewise-affine PHOTO deformation, not recovered physical geometry.
90 degrees uses the source photo as reference, not a measured seat/back angle.
The user supplies a backrest rectangle/hinge; side-view chairs work best.
"""
import cv2
import numpy as np
from src.edit2d import opaque_bbox

BACKREST_ANGLES = (90, 105, 120, 135, 150, 165, 180)


def recline_rgba(rgba, angle=90, hinge=55, left=0, right=100, direction='left'):
    """Return (RGBA, report); report.offset locates output relative to source.

    Above the hinge, x is sheared by sin(recline), y compressed by cos(recline).
    The hinge line is fixed and everything outside the selected mask is untouched.
    A small projected thickness at 180 degrees avoids a singular, invisible layer.
    Region values are percentages of the foreground box in displayed coordinates.
    """
    angle = float(angle)
    if not 90 <= angle <= 180:
        raise ValueError('Backrest angle must be between 90 and 180 degrees.')
    if not (5 <= float(hinge) <= 90 and 0 <= float(left) < float(right) <= 100):
        raise ValueError('Set hinge between 5–90% and left edge below right edge.')
    if direction not in ('left', 'right'):
        raise ValueError('Recline direction must be left or right.')
    h, w = rgba.shape[:2]
    bx0, by0, bx1, by1 = opaque_bbox(rgba[..., 3])
    x0 = int(round(bx0 + (bx1-bx0)*float(left)/100))
    x1 = int(round(bx0 + (bx1-bx0)*float(right)/100))
    hy = int(round(by0 + (by1-by0)*float(hinge)/100))
    mask = np.zeros((h, w), bool)
    mask[by0:hy, x0:x1] = rgba[by0:hy, x0:x1, 3] > 0
    delta = np.deg2rad(angle - 90)
    sign = 1 if direction == 'right' else -1
    shear, squash = -sign*np.sin(delta), max(.045, float(np.cos(delta)))
    M = np.array([[1., shear, -shear*hy], [0., squash, hy*(1-squash)]], np.float64)
    info = dict(angle=angle, hinge=hy, region=(x0, by0, x1, hy), offset=(0, 0),
                M_backrest=M, moving_pixels=int(mask.sum()), direction=direction,
                approximation='guided 2D projection; 90 = source reference')
    if angle == 90 or not mask.any():
        return rgba.copy(), info
    corners = np.array([[x0, by0, 1], [x1, by0, 1], [x0, hy, 1], [x1, hy, 1]]) @ M.T
    ox = min(0, int(np.floor(corners[:, 0].min()))-2)
    oy = min(0, int(np.floor(corners[:, 1].min()))-2)
    ex = max(w, int(np.ceil(corners[:, 0].max()))+2)
    ey = max(h, int(np.ceil(corners[:, 1].max()))+2)
    M_out = M.copy()
    M_out[:, 2] -= [ox, oy]
    f = rgba.astype(np.float32)
    alpha = f[..., 3:4]/255.
    moving = np.concatenate([f[..., :3]*alpha, f[..., 3:4]], axis=2)
    moving[~mask] = 0
    warped = cv2.warpAffine(moving, M_out, (ex-ox, ey-oy), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    fixed = rgba.copy()
    fixed[mask] = 0
    # Blend the moving backrest over the fixed seat/legs in premultiplied space.
    base = np.zeros_like(warped)
    fa = fixed[..., 3:4].astype(np.float32)/255.
    base[-oy:h-oy, -ox:w-ox] = np.concatenate([fixed[..., :3]*fa, fixed[..., 3:4]], axis=2)
    wa = warped[..., 3:4]/255.
    result = warped + base*(1-wa)
    a = result[..., 3:4]
    rgb = result[..., :3]/np.maximum(a/255., 1e-6)
    out = np.clip(np.concatenate([rgb, a], axis=2)+.5, 0, 255).astype(np.uint8)
    # The hinge is a strict boundary: preserve every lower-body source pixel,
    # including antialiased edges and unused RGB under transparent pixels.
    out[hy-oy:h-oy, -ox:w-ox] = rgba[hy:h]
    info.update(offset=(ox, oy), M_backrest=M_out)
    return out, info
