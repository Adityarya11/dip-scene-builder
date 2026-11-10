"""Gradio app for Hugging Face Spaces (CPU).
Loads precomputed artifacts (CLIP image embeddings, cut-outs, palettes) + the CLIP *text* encoder only.

Two stages:
  Build scene    : CLIP retrieval + harmony search + 3D export (heavy, runs once per prompt)
  Apply edits    : cached 2D layers; refresh the small proxy GLB only when a chair pose changes.
                   No CLIP inference, harmony search or segmentation is repeated.
"""
import os
import tempfile
import time
import gradio as gr
import cv2
import numpy as np
import torch
from src.scene3d import export_scene3d
from src.compose import (Library, SCENES, compose, render, render_scene, item_card, slot_labels, default_pair,
                         gap_limits, compare_strip, draw_overlay)
from src.backrest import BACKREST_ANGLES
from src.palette import harmony_score
from src.features import load_clip

cv2.setNumThreads(1)
torch.set_num_threads(2)

# This existing Space requires a registered ZeroGPU function at startup.
# Build/edit paths use CPU only and must not allocate a GPU or consume its quota.
if os.environ.get("SPACE_ID"):
    import spaces
    gpu_decorator = spaces.GPU
else:
    gpu_decorator = lambda func: func


@gpu_decorator
def _reserved_gpu_backend():
    """Startup compatibility only; never called by the CPU scene/editor callbacks."""
    return None

t0 = time.time()
LIB = Library(os.environ.get("ART_DIR", "artifacts"), clip=load_clip("cpu"))
print(f"ready in {time.time() - t0:.1f}s, {len(LIB.meta)} products")

EXAMPLES = ["cozy reading corner", "modern living room", "minimalist home office",
            "rustic dining room", "warm lounge", "elegant entryway"]


# ---------------------------------------------------------------- heavy stage: build the scene once
def run(prompt, harmony):
    started = time.time()
    on = compose(LIB, prompt or "living room", True)
    off_ids = [int(c[0]) for c in on['candidates']]
    off_score = harmony_score([LIB.summ[i] for i in off_ids])
    off_image = render(LIB, off_ids, on['slots'])
    changed = sum(a != b for a, b in zip(off_ids, on['ids']))
    r = on if harmony else dict(on, ids=off_ids, image=off_image,
                               harmony=off_score, harmony_enabled=False)
    glb_path = export_scene3d(LIB, r)
    cards = [(item_card(LIB, i), f"{s['q']} | {LIB.meta.image_id[i]}: {str(LIB.meta.caption[i])[:65]}") for i, s in zip(r["ids"], r["slots"])]
    info = (f"Room: {r['scene']} | Harmony: {'on' if harmony else 'off'} | "
            f"Time: {time.time()-started:.2f}s\n\n"
            f"Colour score: {off_score:.3f} (off) / {on['harmony']:.3f} (on) | "
            f"Products changed: {changed}/{len(off_ids)}")
    if not changed:
        info += "\n\nBoth settings selected the same products."

    # per-session state for the light 2D editing stage (slots are COPIED, the shared SCENES templates stay untouched)
    slots = [dict(s) for s in on["slots"]]
    labels = slot_labels(slots)
    state = dict(ids_on=[int(i) for i in on["ids"]], ids_off=off_ids, slots=slots, labels=labels,
                 harmony=bool(harmony), rot={}, backrest={}, scene=on["scene"],
                 glb_path=glb_path, glb_pose=())
    choices = [(lab, j) for j, lab in enumerate(labels)]
    pair = default_pair(slots)
    chair_choices = [(lab, j) for j, lab in enumerate(labels) if slots[j]['cat'] == 'chair']
    chair = chair_choices[0][1] if chair_choices else None
    if pair is None:
        edit_info = "Spacing needs two objects."
        a_upd = b_upd = gr.update(choices=choices, value=None, interactive=False)
        gap_upd = gr.update(interactive=False)
        on_upd = gr.update(value=False, interactive=False)
    else:
        a_upd = gr.update(choices=choices, value=pair[0], interactive=True)
        b_upd = gr.update(choices=choices, value=pair[1], interactive=True)
        gap_upd = gr.update(interactive=True)
        on_upd = gr.update(value=False, interactive=True)
        edit_info = _limits_text(state, pair[0], pair[1])
    rot_upd = gr.update(choices=chair_choices, value=chair, interactive=bool(chair_choices))
    if not chair_choices:
        edit_info += '\n\nThis layout has no chair. Choose a reading corner, home office or dining room to edit a backrest.'
    png = _save_png(r["image"])
    return (r["image"], cards, info, glb_path, off_image, on['image'],
            state, a_upd, b_upd, on_upd, gap_upd, rot_upd, edit_info, None, "", png,
            gr.update(value=90, interactive=bool(chair_choices)), gr.update(value='outward'))


# ---------------------------------------------------------------- light stage: 2D geometric edits only
def _ids(state):
    return state["ids_on"] if state["harmony"] else state["ids_off"]


def _sync_3d(state):
    """Reuse the GLB for spacing/mask-only edits; rebuild only changed chair poses."""
    pose = tuple(sorted((int(j), float(c['angle']), c.get('direction', 'outward'))
                        for j, c in state.get('backrest', {}).items()))
    if state.get('glb_pose') == pose and state.get('glb_path'):
        return gr.skip()
    path = export_scene3d(LIB, dict(ids=_ids(state), slots=state['slots'], scene=state['scene'],
                                   harmony_enabled=state['harmony'], backrest=state.get('backrest', {})))
    state.update(glb_path=path, glb_pose=pose)
    return path


def _limits_text(state, a, b):
    s = gap_limits(LIB, _ids(state), state["slots"], {}, a, b, backrests=state.get('backrest', {}))
    lab = state["labels"]
    msg = (f"**{lab[a]}** stays fixed, **{lab[b]}** moves horizontally ({s['side']} of {lab[a]}). "
           f"Feasible gap: {s['lo']} to {s['hi']} px. Current gap: {s['current']} px.")
    if not s["fits"]:
        msg += " These two objects cannot be separated inside the canvas at this size."
    return msg


def _save_png(img):
    path = os.path.join(tempfile.mkdtemp(prefix="scene2d_"), "scene_2d.png")
    img.save(path)
    return path


def _matrix(M):
    return np.array2string(np.asarray(M), precision=3, suppress_small=True, floatmode="fixed")


def pair_changed(state, a, b):
    if not state or a is None or b is None or a == b:
        return "Pick two different objects."
    return _limits_text(state, int(a), int(b))


def chair_changed(state, j):
    cfg = (state or {}).get('backrest', {}).get(int(j), {}) if j is not None else {}
    return tuple(cfg.get(k, v) for k, v in dict(angle=90, hinge=55, left=0, right=100, direction='outward').items())


def apply_2d(state, spacing_on, a, b, gap_px, rot_j, angle, show,
             hinge=55, left=0, right=100, direction='outward'):
    """Recline only a selected chair backrest; reuse all retrieval and fixed pixels."""
    if not state:
        return [gr.skip()] * 3 + [gr.skip(), "Build a scene first.", gr.update(visible=False), "", None, gr.skip()]
    t_start = time.perf_counter()
    state = dict(state)
    backrests = dict(state.get('backrest', {}))
    config = dict(angle=float(angle), hinge=float(hinge), left=float(left), right=float(right), direction=direction)
    if rot_j is not None:
        rot_j = int(rot_j)
        if state['slots'][rot_j]['cat'] != 'chair':
            raise gr.Error('Choose a chair to edit its backrest.')
        if not 0 <= left < right <= 100 or not 90 <= angle <= 180:
            raise gr.Error('Backrest angle: 90–180 degrees. Left edge must be smaller than right edge.')
        if int(angle) == 90:
            backrests.pop(rot_j, None)
        else:
            backrests[rot_j] = config
    state['rot'] = {}
    state['backrest'] = backrests
    edits = {'backrest': backrests}
    have_pair = a is not None and b is not None and int(a) != int(b)
    if have_pair:
        edits["gap"] = dict(a=int(a), b=int(b), px=int(gap_px) if spacing_on else None)

    img_on, rep_on = render_scene(LIB, state["ids_on"], state["slots"], 0, edits)
    img_off, rep_off = render_scene(LIB, state["ids_off"], state["slots"], 0, edits)   # same settings for both comparison images
    main_img, rep = (img_on, rep_on) if state["harmony"] else (img_off, rep_off)
    png = _save_png(main_img)                                  # clean image, no overlay, for download
    shown, shown_on, shown_off = main_img, img_on, img_off
    if show:
        shown, shown_on, shown_off = draw_overlay(main_img, rep), draw_overlay(img_on, rep_on), draw_overlay(img_off, rep_off)

    lab = state["labels"]
    lines = ['**Backrest angle updates both the 2D image and 3D chair. Spacing changes the 2D layout only.**']
    g = rep["gap"]
    if g and g["applied"]:
        lines.append(f"- Spacing {lab[g['a']]} (fixed) / {lab[g['b']]} (moved {g['side']}): requested {g['requested']} px, "
                     f"achieved **{g['achieved']} px**. Feasible range {g['lo']} to {g['hi']} px.")
        if not g["exact"]:
            lines.append(f"  - {g['requested']} px does not fit in the canvas, so the nearest feasible gap was used.")
        if not g["fits"]:
            lines.append("  - Even a gap of 0 does not fit; the object is pinned to the canvas edge and the negative value is the overlap.")
    elif g:
        lines.append(f"- Spacing not applied. Current gap {lab[g['a']]} / {lab[g['b']]}: {g['achieved']} px (feasible 0 to {g['hi']} px).")
    if backrests:
        lines.append('- Backrest settings: ' + ', '.join(f"{lab[j]} {cfg['angle']:g}°" for j, cfg in sorted(backrests.items())))
        lines.append('- Only the marked backrest region bends. Seat and legs keep their original position unless spacing moves the chair.')
        for j in sorted(backrests):
            if rep['items'][j]['clipped']:
                lines.append(f'- {lab[j]} backrest reaches outside the scene. Reduce the angle or change its direction.')
            if not rep['items'][j]['backrest']['moving_pixels']:
                lines.append(f'- No foreground pixels in {lab[j]} backrest region. Adjust the region controls.')
    lines.append('90° uses the original photo as reference. This is a guided 2D deformation, not a measured physical recline.')
    cmp_img = None
    mats = ""
    if rot_j is not None:
        it = rep["items"][int(rot_j)]
        detail = it.get('backrest')
        mats += f"{lab[int(rot_j)]}: fixed seat/legs, source -> canvas:\n{_matrix(it['M_net'])}\n\n"
        if detail:
            mats += (f"Backrest-only affine matrix (scaled chair -> expanded layer):\n{_matrix(detail['M_backrest'])}\n"
                     f"Backrest source -> canvas:\n{_matrix(it['M_backrest_net'])}\n"
                     f"Hinge row: {detail['hinge']} px; region: {detail['region']}\n\n")
        else:
            mats += 'Backrest: identity; source photo unchanged at 90 degrees.\n\n'
    if g:
        itb = rep["items"][g["b"]]
        mats += (f"{lab[g['b']]}  net matrix after spacing (translation tx = {itb['M_net'][0, 2]:.1f}):\n{_matrix(itb['M_net'])}\n\n"
                 f"requested gap {g['requested']} px, achieved {g['achieved']} px, "
                 f"box A x:[{rep['items'][g['a']]['bbox'][0]},{rep['items'][g['a']]['bbox'][2]}), "
                 f"box B x:[{itb['bbox'][0]},{itb['bbox'][2]})\n")
    if show and rot_j is not None:
        cmp_img = compare_strip(LIB, _ids(state), state["slots"], int(rot_j), backrest=config)
    glb_update = _sync_3d(state)
    ms = (time.perf_counter() - t_start) * 1000
    lines.append(f"\nEdit applied in {ms:.0f} ms on CPU. No model inference. "
                 + ('3D chair updated.' if isinstance(glb_update, str) else 'Existing 3D file reused.'))
    mats_md = f"```\n{mats}```" if mats else ""
    return (shown, shown_off, shown_on, state, "\n".join(lines), gr.update(value=cmp_img, visible=cmp_img is not None), mats_md, png, glb_update)


def reset_2d(state):
    if not state:
        return [gr.skip()] * 3 + [state, "Build a scene first.", gr.update(visible=False), "", None, gr.skip()]
    state = dict(state, rot={}, backrest={})
    main = render_scene(LIB, _ids(state), state["slots"], 0)[0]
    on = render_scene(LIB, state["ids_on"], state["slots"], 0)[0]
    off = render_scene(LIB, state["ids_off"], state["slots"], 0)[0]
    return main, off, on, state, "Image and chair pose reset.", gr.update(visible=False), "", _save_png(main), _sync_3d(state)


with gr.Blocks(title="Commerce Scene Builder") as demo:
    gr.Markdown("# Commerce Scene Builder\n"
                "Enter a room description to arrange matching products in a scene.\n\n"
                f"Room types: {', '.join(SCENES)}.")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="cozy reading corner", scale=4)
        harmony = gr.Checkbox(label="Colour harmony", value=True, scale=1)
        btn = gr.Button("Build scene", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=prompt)
    info = gr.Markdown()
    out = gr.Image(label="Composed scene", type="pil", format="png")
    png_file = gr.File(label="Download the 2D scene (PNG)")
    state = gr.State(None)

    with gr.Group():
        gr.Markdown("### Scene edits\n"
                    "Set 2D spacing or recline the chair backrest. The same angle updates the 3D backrest and headrest. "
                    "The photo edit is a guided 2D approximation; 90° keeps the source photo.")
        with gr.Row():
            spacing_on = gr.Checkbox(label="Set spacing", value=False)
            dd_a = gr.Dropdown(label="Object 1 (fixed)", choices=[], interactive=False)
            dd_b = gr.Dropdown(label="Object 2 (moves horizontally)", choices=[], interactive=False)
            gap_px = gr.Slider(0, 768, value=120, step=1, label="Horizontal gap between opaque boxes (px)")
        with gr.Row():
            dd_rot = gr.Dropdown(label="Chair backrest", choices=[], interactive=False)
            angle = gr.Radio(list(BACKREST_ANGLES), value=90, label="Backrest angle (90° = source photo)")
            show = gr.Checkbox(label="Show transformations", value=False)
        with gr.Accordion('Adjust backrest region', open=False):
            gr.Markdown('Use Show transformations to check the cyan region and orange hinge line. '
                        'Adjust these to include the backrest, not the arms or seat. Side-view chairs work best.')
            with gr.Row():
                hinge = gr.Slider(5, 90, value=55, step=1, label='Hinge height (% from top)')
                left = gr.Slider(0, 95, value=0, step=1, label='Backrest left edge (%)')
                right = gr.Slider(5, 100, value=100, step=1, label='Backrest right edge (%)')
                direction = gr.Radio(['outward', 'inward', 'left', 'right'], value='outward', label='Recline direction')
        with gr.Row():
            apply_btn = gr.Button("Apply changes", variant="primary")
            reset_btn = gr.Button("Reset edits")
        edit_info = gr.Markdown()
        cmp_img = gr.Image(label="Original cut-out to transformed cut-out", type="pil", visible=False, format="png")
        with gr.Accordion("Backrest / base matrices and measured gap", open=False):
            mats = gr.Markdown()

    gal = gr.Gallery(label="Selected products and colours", columns=5, height="auto")
    with gr.Tabs():
        with gr.Tab("Harmony comparison"):
            with gr.Row():
                off_out = gr.Image(label="Harmony off", type="pil")
                on_out = gr.Image(label="Harmony on", type="pil")
        with gr.Tab("3D layout"):
            gr.Markdown("Drag to rotate. Scroll to zoom. The layout uses simplified shapes and product colours.")
            out3d = gr.Model3D(label="Room layout (GLB)", display_mode="solid", height=550,
                              camera_position=(45, 65, 11), clear_color=(0.95, 0.95, 0.94, 1.0))
    gr.Markdown("Click Build scene again after changing the colour harmony setting.")
    outputs = [out, gal, info, out3d, off_out, on_out, state, dd_a, dd_b, spacing_on, gap_px, dd_rot,
               edit_info, cmp_img, mats, png_file, angle, direction]
    btn.click(run, [prompt, harmony], outputs, concurrency_limit=1, api_name="build_scene")
    prompt.submit(run, [prompt, harmony], outputs, concurrency_limit=1, api_name=False)

    edit_out = [out, off_out, on_out, state, edit_info, cmp_img, mats, png_file, out3d]
    apply_btn.click(apply_2d, [state, spacing_on, dd_a, dd_b, gap_px, dd_rot, angle, show, hinge, left, right, direction], edit_out,
                    api_name="apply_2d_changes")
    reset_btn.click(reset_2d, [state], edit_out, api_name=False).then(
        lambda: (gr.update(value=90), gr.update(value='outward')), outputs=[angle, direction], api_name=False)
    dd_rot.input(chair_changed, [state, dd_rot], [angle, hinge, left, right, direction], api_name=False)
    dd_a.input(pair_changed, [state, dd_a, dd_b], edit_info, api_name=False)
    dd_b.input(pair_changed, [state, dd_a, dd_b], edit_info, api_name=False)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
