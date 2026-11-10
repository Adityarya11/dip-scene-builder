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

    # Keep source swatches unchanged; derived shades are structural only.
    if np.linalg.norm(secondary-primary) < 35:
        secondary = _mix(primary, [42, 43, 45], .23)

    light = _mix(primary, [255, 255, 255], .16)
    dark = _mix(primary, [32, 34, 37], .36)
    accent = secondary.copy()
    if np.linalg.norm(accent-primary) < 35:
        accent = _mix(primary, [32, 34, 37], .42)
    return tuple(np.clip(c, 0, 255) for c in (primary, secondary, light, dark, accent))


def export_scene3d(lib, result, output_path=None):
    scene = trimesh.Scene()
    template = str(result.get('scene', 'living room')).strip().lower()
    product_indices = [int(i) for i in result['ids']]
    product_ids = [str(lib.meta.iloc[i]['image_id']) for i in product_indices]
    # A numeric harmony score is not the harmony-enabled selection flag.
    harmony_enabled = result.get('harmony_enabled')
    scene.metadata.update({
        'representation': 'category-based product-coloured proxies; not product reconstructions',
        'scene': template,
        'product_ids': product_ids,
        'product_indices': product_indices,
        'categories': [slot['cat'] for slot in result['slots']],
        'harmony_enabled': None if harmony_enabled is None else bool(harmony_enabled),
        'layout_version': 2,
    })
    item_transform = np.eye(4)

    def rgba(rgb, shade=1.0):
        a = np.asarray(rgb, dtype=float).ravel()[:3]
        if len(a) != 3:
            a = np.array([170., 150., 130.])
        return np.r_[np.clip(a * shade, 0, 255).astype('uint8'), 255]

    def box(name, size, pos, rgb, shade=1):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(pos)
        mesh.apply_transform(item_transform)
        mesh.visual.face_colors = rgba(rgb, shade)
        scene.add_geometry(mesh, node_name=name)

    def cylinder(name, radius, height, pos, rgb, shade=1, sections=20):
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
        mesh.apply_translation(pos)
        mesh.apply_transform(item_transform)
        mesh.visual.face_colors = rgba(rgb, shade)
        scene.add_geometry(mesh, node_name=name)

    def sphere(name, radius, pos, rgb):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        mesh.apply_translation(pos)
        mesh.apply_transform(item_transform)
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

    # Metre-scale floor plans: x, y, rotation about Z (not canvas positions).
    layouts = {
        'living room': {
            'rug': [(0, -.25, 0)], 'sofa': [(0, 1.45, 0)],
            'chair': [(-2.1, -.15, np.pi/2)], 'table': [(0, -.35, 0)],
            'lamp': [(1.75, 1.55, 0)], 'shelf': [(-2.8, 2.35, 0)],
            'plant_vase': [(-2.65, .8, 0)], 'wall_art': [(0, 2.66, 0)]},
        'reading corner': {
            'rug': [(-.4, -.35, 0)], 'chair': [(-.6, .15, -.18)],
            'shelf': [(-2.7, 2.35, 0)], 'lamp': [(.65, .6, 0)],
            'plant_vase': [(2.3, 1.7, 0)], 'table': [(.85, -.5, 0)]},
        'home office': {
            'table': [(.1, 1.25, 0)], 'chair': [(.1, -.1, np.pi)],
            'shelf': [(-2.7, 2.35, 0)], 'lamp': [(.65, 1.4, 0)],
            'plant_vase': [(2.75, .6, 0)], 'wall_art': [(.1, 2.66, 0)]},
        'dining room': {
            'table': [(0, 0, 0)], 'rug': [(0, 0, 0)],
            'chair': [(-1.4, 0, np.pi/2), (1.4, 0, -np.pi/2),
                      (0, 1.12, 0), (0, -1.12, np.pi)],
            'wall_art': [(0, 2.66, 0)]},
        'bedroom': {
            # This template retrieves an accent chair/dresser, not a bed.
            'rug': [(-.35, -.6, 0)], 'chair': [(-1.9, .55, -.25)],
            'shelf': [(1.2, 2.25, 0)], 'plant_vase': [(2.8, 1.2, 0)],
            'wall_art': [(1.2, 2.66, 0)]},
        'entryway': {
            'shelf': [(0, 2.25, 0)], 'wall_art': [(0, 2.66, 0)],
            'lamp': [(-1.7, 1.95, 0)], 'plant_vase': [(1.9, 1.8, 0)],
            'rug': [(0, -.25, 0)]},
    }
    positions = layouts.get(template, layouts['living room'])
    counts = {}
    for index, slot in zip(result['ids'], result['slots']):
        cat = slot['cat']
        count = counts.get(cat, 0)
        counts[cat] = count + 1
        poses = positions.get(cat, layouts['living room'].get(cat, [(0., 0., 0.)]))
        x, y, angle = poses[count % len(poses)]
        # Additional instances get full-width separation rather than a tiny nudge.
        repeat = count // len(poses)
        x += (1.35 if cat == 'wall_art' else 1.65) * repeat
        item_transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1], [x, y, 0])
        rgb, secondary, light, dark, accent = _product_colors(lib, index)
        name = f'{cat}_{count}'
        if __import__('os').environ.get('DIP_DEBUG_PALETTES') == '1':
            print(f'[3D palette] {name} product_index={index} primary={rgb.astype(int).tolist()} secondary={secondary.astype(int).tolist()}')

        if cat == 'rug':
            rw, rd = (2.0, 3.4) if template == 'entryway' else (3.71, 2.70)
            box(name+'_rug_edge', (rw, rd, .014), (x, y, .012), dark)
            box(name+'_rug', (rw-.20, rd-.19, .028), (x, y, .032), rgb)
            # Inset stripes echo the selected rug palette.
            for j, offset in enumerate((-rd/2+.4, rd/2-.4)):
                box(f'{name}_stripe_{j}', (rw-.37, .075, .006), (x, y+offset, .049), accent)

        elif cat in ('chair', 'sofa'):
            if cat == 'chair' and template in ('dining room', 'home office'):
                # Upright dining/task chair rather than a full-size armchair.
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        box(f'{name}_leg_{sx}_{sy}', (.065, .065, .44),
                            (x+sx*.24, y+sy*.23, .22), dark)
                box(name+'_seat', (.62, .62, .12), (x, y, .50), rgb)
                box(name+'_back', (.62, .10, .58), (x, y+.27, .83), secondary)
                continue
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
            tw, td, th = {'dining room': (1.85, 1.10, .79),
                          'home office': (1.75, .85, .79)}.get(template, (1.25, .76, .48))
            box(name+'_top', (tw, td, .10), (x, y, th-.05), rgb)
            for sx in (-1, 1):
                for sy in (-1, 1):
                    box(f'{name}_leg_{sx}_{sy}', (.08, .08, th-.10),
                        (x+sx*(tw/2-.12), y+sy*(td/2-.12), (th-.10)/2), dark)

        elif cat == 'lamp':
            if template == 'home office':
                # The office slot requests a desk lamp: sit it on the desktop.
                cylinder(name+'_foot', .14, .05, (x, y, .815), dark)
                cylinder(name+'_stem', .024, .38, (x, y, 1.03), secondary)
                cylinder(name+'_shade', .18, .18, (x, y, 1.29), rgb)
                continue
            cylinder(name+'_foot', .23, .07, (x, y, .035), [75, 75, 73])
            cylinder(name+'_stem', .032, 1.48, (x, y, .77), [94, 87, 80])
            cylinder(name+'_shade', .31, .35, (x, y, 1.66), rgb)
            sphere(name+'_bulb', .11, (x, y, 1.46), [254, 238, 194])

        elif cat == 'shelf':
            if template in ('bedroom', 'entryway'):
                # Dresser/console slots should not become towering bookcases.
                for side in (-1, 1):
                    box(f'{name}_foot_{side}', (.14, .40, .18), (x+side*.69, y, .09), dark)
                box(name+'_cabinet', (1.75, .55, .80), (x, y, .58), rgb)
                box(name+'_top', (1.85, .61, .08), (x, y, 1.02), secondary)
                for side in (-1, 1):
                    box(f'{name}_door_{side}', (.80, .035, .65),
                        (x+side*.42, y-.29, .59), secondary)
                    box(f'{name}_handle_{side}', (.09, .04, .04),
                        (x+side*.12, y-.32, .65), dark)
                continue
            # Respect the template's shelf placement.
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
                    (rgb, secondary, light, dark, accent)[j])

        elif cat == 'plant_vase':
            cylinder(name+'_pot', .26, .43, (x, y, .215), accent)
            cylinder(name+'_trunk', .045, .72, (x, y, .79), [85, 101, 50])
            for j, (dx, dy, dz, r) in enumerate([
                (-.20, 0, 1.05, .24),(.22,.04,1.17,.24),(.02,-.22,1.32,.25),
                (.04,.18,1.41,.25),(-.08,0,1.57,.23)]):
                sphere(f'{name}_leaf_{j}', r, (x+dx,y+dy,dz), [75, 142 + (j%2)*15, 76])

        elif cat == 'wall_art':
            box(name+'_frame', (1.20, .075, .90), (x, y+.03, 1.90), dark)
            box(name+'_art', (1.08, .016, .78), (x, y-.02, 1.90), rgb)

    # Critical fix: GLB/glTF viewers normally use Y-up, while trimesh scenes here use Z-up.
    # Rotation -90 degrees around X maps (x,y,z) -> (x,z,-y).
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))

    if output_path is None:
        output_path = Path(tempfile.mkdtemp(prefix='dip_scene3d_')) / 'room.glb'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(output_path), file_type='glb')
    return str(output_path)
