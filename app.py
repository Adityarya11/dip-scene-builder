"""Gradio app for Hugging Face Spaces (CPU).
Loads precomputed artifacts (CLIP image embeddings, cut-outs, palettes) + the CLIP *text* encoder only."""
import os
import time
import gradio as gr
from src.scene3d import export_scene3d
from src.compose import Library, SCENES, compose, item_card
from src.features import load_clip

import os

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
    r = compose(LIB, prompt or "living room", harmony)
    glb_path = export_scene3d(LIB, r)
    cards = [(item_card(LIB, i), f"{s['q']}: {LIB.meta.caption[i][:70]}") for i, s in zip(r["ids"], r["slots"])]
    info = (f"Scene template: **{r['scene']}**  |  style words: *{r['style'] or '-'}*  |  "
            f"colour-harmony score: **{r['harmony']:.2f}** ({'harmony ON' if harmony else 'harmony OFF'})  |  "
            f"built in {r['seconds']:.2f}s")
    return r["image"], cards, info, glb_path


with gr.Blocks(title="AI-Assisted Commerce Scene Builder") as demo:
    gr.Markdown("# AI-Assisted Commerce Scene Builder (2.5D)\n"
                "Type a room idea. CLIP retrieves matching Amazon product photos, classical DIP + U2-Net cut them out, "
                "k-means palettes pick a colour-harmonious set, and the items are composed into a room.\n\n"
                f"Known scene types: {', '.join(SCENES)}.")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="cozy reading corner", scale=4)
        harmony = gr.Checkbox(label="Colour harmony ON", value=True, scale=1)
        btn = gr.Button("Build scene", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=prompt)
    info = gr.Markdown()
    out = gr.Image(label="Composed scene", type="pil")
    gal = gr.Gallery(label="Retrieved items + dominant-colour palettes", columns=5, height="auto")
    out3d = gr.Model3D(
    label="Interactive 3D Room",
    display_mode="solid",
    height=550,
    camera_position=(45, 65, 11),
    clear_color=(0.93, 0.93, 0.95, 1.0)
    )
    btn.click(run, [prompt, harmony], [out, gal, info, out3d])
    prompt.submit(run, [prompt, harmony], [out, gal, info, out3d])

if __name__ == "__main__":
    demo.launch()
