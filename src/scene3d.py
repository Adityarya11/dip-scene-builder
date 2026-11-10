"""Lightweight, rotatable 3D room preview for DIP Scene Builder.

Meshes are category-based proxies coloured from selected product palettes,
NOT 3D reconstructions of the original product photographs.
All geometry is made on CPU using trimesh and saved as a Y-up GLB.
"""
from pathlib import Path
import tempfile
import numpy as np
import colorsys
import trimesh


def _mix(a, b, amount):
    """Mix two 8-bit RGB colours without changing their hue arbitrarily."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.clip(a * (1 - amount) + b * amount, 0, 255)


def _tone(rgb, light=0, saturation=1.0):
    """Shift brightness/saturation modestly, preserving original product hue."""
    r, g, b = np.clip(np.asarray(rgb, float) / 255, 0, 1)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = min(.80, s * saturation)
    v = np.clip(v + light, .12, .94)
    return np.asarray(colorsys.hsv_to_rgb(h, s, v)) * 255


def _product_colors(lib, idx):
    """Use actual selected-product swatches; avoid white backgrounds and duplicate tones.

    Product colours still may be neutral if the source product is neutral.
    The largest vivid swatch is favoured, rather than artificial recolouring.
    """
    sw = np.asarray(lib.swatch(idx, 5), dtype=float).reshape(-1, 3)
    sw = sw[np.isfinite(sw).all(axis=1)]
    if len(sw) == 0:
        sw = np.array([[145., 108., 82.]])
    if sw.max() <= 1.01:
        sw *= 255
    sw = np.clip(sw, 0, 255)

    hsv = np.asarray([colorsys.rgb_to_hsv(*(c / 255.0)) for c in sw])
    sat, val = hsv[:, 1], hsv[:, 2]
    # Disfavor white/very bright background patches; keep legitimate neutral products.
    background = (val > .87) & (sat < .16)
    eligible = np.where(~background)[0]
    if not len(eligible):
        eligible = np.arange(len(sw))
    # Earlier palette positions are typically the larger clusters.
    ranked = sorted(eligible, key=lambda j: (-(.70 * sat[j] + .30 * (1 - abs(val[j] - .54)) - .065*j)))
    primary = sw[ranked[0]].copy()
    # Pick a genuinely different second swatch; avoid near-white secondary swatches.
    alternatives = [j for j in eligible if j != ranked[0]]
    if alternatives:
        j = max(alternatives, key=lambda j: np.linalg.norm(sw[j] - primary) - 6*j)
        secondary = sw[j].copy()
    else:
        secondary = primary.copy()

    # Moderate shade differences without changing the selected product's hue.
    primary = _tone(primary, saturation=1.35)
    if np.linalg.norm(secondary-primary) < 35:
        secondary = _mix(primary, [42, 43, 45], .23)
    else:
        secondary = _tone(secondary, saturation=1.25)
    light = _mix(primary, [255, 255, 255], .16)
    dark = _mix(primary, [32, 34, 37], .36)
    accent = _tone(secondary, saturation=1.35)
    if np.linalg.norm(accent-primary) < 35:
        accent = _mix(primary, [32, 34, 37], .42)
    return tuple(np.clip(c, 0, 255) for c in (primary, secondary, light, dark, accent))


def export_scene3d(lib, result, output_path=None):
    scene = trimesh.Scene()

    def rgba(rgb, shade=1.0):
        a = np.asarray(rgb, dtype=float).ravel()[:3]
        if len(a) != 3:
            a = np.array([170., 150., 130.])
        return np.r_[np.clip(a * shade, 0, 255).astype('uint8'), 255]

    def box(name, size, pos, rgb, shade=1):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(pos)
        mesh.visual.face_colors = rgba(rgb, shade)
        scene.add_geometry(mesh, node_name=name)

    def cylinder(name, radius, height, pos, rgb, shade=1, sections=20):
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        mesh.apply_translation(pos)
        mesh.visual.face_colors = rgba(rgb, shade)
        scene.add_geometry(mesh, node_name=name)

    def sphere(name, radius, pos, rgb):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        mesh.apply_translation(pos)
        mesh.visual.face_colors = rgba(rgb)
        scene.add_geometry(mesh, node_name=name)

    # Working coordinates: x=left/right, y=front/back, z=height.
    # At export we convert Z-up to the Y-up GLB convention.
    wood = [151, 116, 84]
    wall = [232, 231, 224]
    box('wood_floor', (7.6, 5.7, .12), (0, 0, -.07), wood)
    box('back_wall', (7.6, .09, 2.9), (0, 2.81, 1.45), wall)
    box('left_wall', (.09, 5.7, 2.9), (-3.76, 0, 1.45), [223, 225, 221])
    # baseboard trim gives much clearer room depth
    box('baseboard_back', (7.55, .08, .14), (0, 2.73, .08), [250, 246, 239])
    box('baseboard_left', (.08, 5.5, .14), (-3.70, -.02, .08), [250, 246, 239])

    # Make room layout independent of 2D screen coordinates; those are not 3D positions.
    # Multiple instances of a category are offset slightly rather than overlapping exactly.
    positions = {
        'rug': (0.0, -.50), 'chair': (-.75, -.65), 'sofa': (-.85, -.25),
        'table': (1.05, -.70), 'lamp': (1.65, .85), 'shelf': (-2.75, 1.95),
        'plant_vase': (2.70, 1.45), 'wall_art': (0.90, 2.66)
    }
    counts = {}
    categories = []
    for index, slot in zip(result['ids'], result['slots']):
        cat = slot['cat']
        categories.append(cat)
        count = counts.get(cat, 0)
        counts[cat] = count + 1
        x, y = positions.get(cat, (0., 0.))
        x += .40 * (count % 3)
        y -= .38 * (count // 3)
        rgb, secondary, light, dark, accent = _product_colors(lib, index)
        name = f'{cat}_{count}'
        if __import__('os').environ.get('DIP_DEBUG_PALETTES') == '1':
            print(f'[3D palette] {name} product_index={index} primary={rgb.astype(int).tolist()} secondary={secondary.astype(int).tolist()}')

        if cat == 'rug':
            box(name+'_rug_edge', (3.71, 2.70, .014), (-.10, -.58, .012), dark)
            box(name+'_rug', (3.51, 2.51, .028), (-.10, -.58, .032), rgb)
            # Inset stripes echo the selected rug palette.
            for j, offset in enumerate((-.94, .94)):
                box(f'{name}_stripe_{j}', (3.34, .075, .006), (-.10, -.58+offset, .049), accent)

        elif cat in ('chair', 'sofa'):
            w = 2.05 if cat == 'sofa' else 1.06
            d = .98
            # All supports touch the floor; coherent seat, back, and arm proportions.
            for sx in (-1, 1):
                for sy in (-1, 1):
                    box(f'{name}_leg_{sx}_{sy}', (.12, .12, .43),
                        (x + sx*(w/2-.17), y + sy*.33, .215), dark)
            box(name+'_seat_base', (w, d, .22), (x, y, .55), secondary)
            box(name+'_seat_cushion', (w-.16, d-.12, .18), (x, y-.035, .74), rgb)
            box(name+'_back', (w, .20, 1.08), (x, y+.39, 1.18), secondary)
            box(name+'_back_cushion', (w-.17, .16, .73), (x, y+.25, 1.18), rgb)
            for side in (-1, 1):
                box(f'{name}_arm_{side}', (.16, .95, .47),
                    (x+side*(w/2-.08), y, .86), secondary)

        elif cat == 'table':
            box(name+'_top', (1.12, .76, .13), (x, y, .76), rgb)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    box(f'{name}_leg_{sx}_{sy}', (.08, .08, .70),
                        (x+sx*.46, y+sy*.28, .35), dark)

        elif cat == 'lamp':
            cylinder(name+'_foot', .23, .07, (x, y, .035), [75, 75, 73])
            cylinder(name+'_stem', .032, 1.48, (x, y, .77), [94, 87, 80])
            cylinder(name+'_shade', .31, .35, (x, y, 1.66), rgb)
            sphere(name+'_bulb', .11, (x, y, 1.46), [254, 238, 194])

        elif cat == 'shelf':
            # Place against back/side wall, not hanging in the air.
            x, y = -3.08, 2.39
            for side in (-1, 1):
                box(f'{name}_side_{side}', (.10, .42, 2.00),
                    (x+side*.53, y, 1.0), dark)
            box(name+'_back', (1.16, .07, 2.0), (x, y+.19, 1.0), dark)
            for j, z in enumerate((.08, .58, 1.10, 1.63, 1.96)):
                box(f'{name}_shelf_{j}', (1.16, .46, .10), (x, y, z), secondary)
            # few book spines make it recognizably a bookshelf
            for j in range(5):
                box(f'{name}_book_{j}', (.10, .26, .31 + j*.035),
                    (x-.40+j*.18, y, .79 + j*.017),
                    [[163,97,87],[100,137,163],[186,155,94],[117,150,124],[112,104,143]][j])

        elif cat == 'plant_vase':
            cylinder(name+'_pot', .26, .43, (x, y, .215), accent)
            cylinder(name+'_trunk', .045, .72, (x, y, .79), [85, 101, 50])
            for j, (dx, dy, dz, r) in enumerate([
                (-.20, 0, 1.05, .24),(.22,.04,1.17,.24),(.02,-.22,1.32,.25),
                (.04,.18,1.41,.25),(-.08,0,1.57,.23)]):
                sphere(f'{name}_leaf_{j}', r, (x+dx,y+dy,dz), [75, 142 + (j%2)*15, 76])

        elif cat == 'wall_art':
            x = min(2.2, max(-1.4, x))
            box(name+'_frame', (1.20, .075, .90), (x, 2.69, 1.90), [91, 66, 45])
            box(name+'_art', (1.08, .016, .78), (x, 2.64, 1.90), rgb)

    # Critical fix: GLB/glTF viewers normally use Y-up, while trimesh scenes here use Z-up.
    # Rotation -90 degrees around X maps (x,y,z) -> (x,z,-y).
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    if output_path is None:
        output_path = Path(tempfile.mkdtemp(prefix='dip_scene3d_')) / 'room.glb'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path), file_type='glb')
    return str(output_path)
