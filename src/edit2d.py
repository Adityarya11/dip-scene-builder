"""Interactive 2D geometric edits of the generated scene (DIP: affine transform, interpolation, alpha).

Everything here works on the cached cut-out layers of the 2D image. It does NOT touch CLIP, the harmony
search, segmentation or the 3D export. "Rotation" is an image-plane rotation of a flat cut-out, NOT a new
viewpoint of the chair and not a 3D reconstruction.

Three small ideas, each one a classic DIP operation:
  1. rotate_rgba   - affine rotation with cv2.getRotationMatrix2D + cv2.warpAffine (bilinear interpolation)
                     on PREMULTIPLIED alpha, so transparent pixels cannot bleed dark/white halos.
  2. opaque_bbox   - bounding box of the opaque foreground (alpha > 128), ignoring transparent borders.
  3. solve_gap     - horizontal translation of object B so the edge-to-edge gap to object A is a given
                     number of pixels, limited by the canvas.
"""
import cv2
import numpy as np

ANGLES = (0, 45, 90, 135, 180)         # degrees, counter-clockwise (OpenCV convention)
OPAQUE = 128                           # alpha above this counts as foreground


def opaque_bbox(alpha, thr=OPAQUE):
    """Half-open bounding box (x0, y0, x1, y1) of pixels with alpha > thr. Width = x1 - x0."""
    ys, xs = np.where(alpha > thr)
    if not len(ys):
        return 0, 0, alpha.shape[1], alpha.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def rotate_rgba(rgba, angle):
    """Rotate an RGBA uint8 image by `angle` degrees (counter-clockwise) about its centre.

    * M = getRotationMatrix2D(centre, angle, scale=1) is a 2x3 affine matrix [R | t].
    * The output rectangle is enlarged to the rotated bounding rectangle (|cos|,|sin| formula) and the
      translation column is shifted, so no part of the cut-out is cropped.
    * Colour is multiplied by alpha BEFORE interpolation (premultiplied alpha) and divided afterwards. Without
      this, bilinear interpolation would mix the (arbitrary) colour of transparent pixels into the edge.
    * Empty margins are trimmed afterwards and the translation in M is adjusted to match.
    Returns (rgba_out, M) where M maps source pixel coordinates to output pixel coordinates.
    """
    h, w = rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(np.ceil(h * s + w * c)), int(np.ceil(h * c + w * s))
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0

    f = rgba.astype(np.float32)
    a = f[..., 3:4] / 255.0
    premult = np.concatenate([f[..., :3] * a, f[..., 3:4]], axis=2)            # (R*a, G*a, B*a, A)
    out = cv2.warpAffine(premult, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha = out[..., 3:4]
    rgb = np.where(alpha > 0, out[..., :3] / np.maximum(alpha / 255.0, 1e-6), 0.0)  # un-premultiply
    res = np.clip(np.concatenate([rgb, alpha], axis=2) + 0.5, 0, 255).astype(np.uint8)

    ys, xs = np.where(res[..., 3] > 0)                                         # trim empty margins
    if len(ys):
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        res = np.ascontiguousarray(res[y0:y1, x0:x1])
        M = M.copy()
        M[0, 2] -= x0
        M[1, 2] -= y0
    return res, M


def to3(M):
    """2x3 affine -> 3x3 homogeneous matrix."""
    return np.vstack([np.asarray(M, np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])


def affine_chain(src_size, flipped, scaled_size, M_rot, origin):
    """Net 2x3 matrix: source cut-out pixel -> canvas pixel = Translate * Rotate * Scale * Flip.
    (continuous pixel coordinates; the +-0.5 pixel-centre convention of resizing is ignored)."""
    w0, h0 = src_size
    w, h = scaled_size
    F = np.eye(3)
    if flipped:
        F[0, 0], F[0, 2] = -1.0, w0 - 1.0
    S = np.diag([w / w0, h / h0, 1.0])
    R = to3(M_rot) if M_rot is not None else np.eye(3)
    T = np.eye(3)
    T[0, 2], T[1, 2] = origin
    return (T @ R @ S @ F)[:2]


def solve_gap(bbox_a, bbox_b, requested, canvas_w):
    """Place B so that the horizontal edge-to-edge gap between the opaque bounding boxes equals `requested`.

    A stays fixed. B keeps its existing side: if B's centre is right of A's centre, B.left = A.right + gap,
    otherwise B.right = A.left - gap. B is only translated horizontally.
    Feasible gaps are 0 <= gap <= hi, where hi keeps B's box inside [0, canvas_w].
    If requested is outside that range the nearest feasible gap is used (achieved != requested).
    If even gap 0 does not fit (hi < 0), B is pinned to the canvas edge and the negative `achieved` is the overlap.
    """
    wb = bbox_b[2] - bbox_b[0]
    right = (bbox_b[0] + bbox_b[2]) >= (bbox_a[0] + bbox_a[2])
    if right:
        current = bbox_b[0] - bbox_a[2]
        hi = canvas_w - wb - bbox_a[2]
    else:
        current = bbox_a[0] - bbox_b[2]
        hi = bbox_a[0] - wb
    achieved = int(min(max(requested, 0), hi)) if hi >= 0 else int(hi)
    left_b = bbox_a[2] + achieved if right else bbox_a[0] - achieved - wb
    return dict(side="right" if right else "left", requested=int(requested), achieved=achieved,
                lo=0, hi=int(hi), current=int(current), dx=int(left_b - bbox_b[0]),
                exact=achieved == int(requested), fits=hi >= 0)
