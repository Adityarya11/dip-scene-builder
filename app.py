"""Gradio app for Hugging Face Spaces (CPU).
Loads precomputed artifacts (CLIP image embeddings, cut-outs, palettes) + the CLIP *text* encoder only."""
import os
import time
import gradio as gr
import cv2
import torch
from src.scene3d import export_scene3d
from src.compose import Library, SCENES, compose, render, item_card
from src.palette import harmony_score
from src.features import load_clip

cv2.setNumThreads(1)
torch.set_num_threads(2)

# This existing Space is configured as ZeroGPU: its startup validator requires
# at least one decorated callback even though this small model runs on CPU.
if os.environ.get("SPACE_ID"):
    import spaces
    gpu_decorator = spaces.GPU
else:
    gpu_decorator = lambda func: func

t0 = time.time()
LIB = Library(os.environ.get("ART_DIR", "artifacts"), clip=load_clip("cpu"))
print(f"ready in {time.time() - t0:.1f}s, {len(LIB.meta)} products")

EXAMPLES = ["cozy reading corner", "modern living room", "minimalist home office",
            "rustic dining room", "bright bedroom", "elegant entryway"]

@gpu_decorator
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
    return r["image"], cards, info, glb_path, off_image, on['image']


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
    out = gr.Image(label="Composed scene", type="pil")
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
    outputs = [out, gal, info, out3d, off_out, on_out]
    btn.click(run, [prompt, harmony], outputs, concurrency_limit=1, api_name="build_scene")
    prompt.submit(run, [prompt, harmony], outputs, concurrency_limit=1, api_name=False)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
