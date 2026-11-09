# AI-Assisted Commerce Scene Builder from Product Images and Text (2.5D)

*Draft. Numbers marked **[fill]** come from `results/metrics.json`, `results/ndcg_table.csv`, `results/iou.json`
and `results/scene_eval.csv` after the full Colab run. Do not quote numbers that are not in those files.*

## 1. Introduction

Online shoppers see products one at a time on white backgrounds and have to imagine how they fit together in a room.
We build a small system that turns a short text prompt ("cozy reading corner") into a composed room image built from real
catalogue photos. The system retrieves matching products, cuts them out, chooses a colour-harmonious set, and places them
in a procedural room using simple placement rules, colour matching and shadows.

The project is a Digital Image Processing exercise: most steps are classical (filtering, histogram equalisation, colour
spaces, thresholding, morphology, clustering, compositing). Deep models are used where they clearly help (CLIP for
text-image matching, U2-Net as a strong segmentation reference) and are compared against classical baselines.

**Scope.** 2.5D only: flat RGBA cut-outs on a fixed-camera room. Real 3D (ABO GLB models) is future work.

## 2. Related work

* **Dataset.** Amazon Berkeley Objects (Collins et al., CVPR 2022) provides product listings, catalogue images and 3D models.
  We use only listing metadata and the small images.
* **Vision-language retrieval.** CLIP (Radford et al., 2021) embeds images and text in one space; we use OpenCLIP ViT-B/32.
* **Foreground segmentation.** Otsu thresholding, GrabCut (Rother et al., 2004) and deep salient-object models, here U2-Net (Qin et al., 2020) via `rembg`.
* **Hand-crafted descriptors.** HSV colour histograms, HOG (Dalal and Triggs, 2005), LBP (Ojala et al., 2002).
* **Colour.** Gray-world white balance, CLAHE (Zuiderveld, 1994), Reinhard colour transfer (2001), colour harmony rules
  (analogous / complementary / triadic hues).
* **Scene generation.** Learned generators (diffusion, NeRF-style) are out of scope; we use deterministic compositing,
  which is explainable and runs on a free CPU.

## 3. Method (mapped to DIP topics)

| Stage | Method | DIP topic |
|---|---|---|
| 3.1 Data | ~190 products x 8 categories from ABO small images | acquisition, resolution |
| 3.2 Pre-processing | resize (bicubic/area), median + Gaussian, gray-world WB (foreground pixels, half strength), CLAHE on L of LAB | interpolation, spatial filtering, colour constancy, histogram equalisation |
| 3.3 Segmentation (a) | distance to border colour -> Otsu -> opening/closing -> largest component -> hole fill; GrabCut if mask is implausible (<8 % or >90 % of image) | thresholding, morphology, graph cuts |
| 3.3 Segmentation (b) | rembg / U2-Net | deep segmentation |
| 3.4 Features | HSV histogram (8x4x4), HOG (128 px), LBP uniform 4x4 grid, CLIP ViT-B/32 | colour, shape, texture descriptors |
| 3.5 Retrieval | cosine similarity over L2-normalised vectors, same `Index` class for all features | similarity search |
| 3.6 Palettes | k-means in LAB, k in 3..5 chosen by silhouette | colour spaces, clustering |
| 3.7 Harmony | per item: circular-mean hue of chromatic clusters (neutral if chroma < 18 or < 15 % coloured). Pair score = 0.7 hue (analogous / complementary / triadic templates) + 0.3 chroma agreement. Scene score = mean over pairs | colour theory |
| 3.8 Composition | prompt -> slot template; top-5 per slot by CLIP text score; brute-force best of 5^n combinations with score = mean normalised CLIP + 3 x harmony | - |
| 3.9 Rendering | placement rules (floor / wall / rug, size priors per category, flip chairs to face the room), Reinhard colour match at 25 %, elliptical Gaussian-blurred contact shadows, alpha blending, procedural gradient wall and plank floor | geometric transforms, colour transfer, blurring, alpha compositing |

Key design choices, each explainable in one sentence:

* Gray-world uses only non-white pixels, otherwise the white background dominates the mean.
* Colour transfer is limited to 25 % because full transfer repaints the product and defeats the purpose of a catalogue image.
* The harmony term is added to the CLIP term, so with harmony OFF the system returns the top-1 item per slot.

## 4. Experiments

* **Data.** 8 categories (sofa, chair, table, lamp, rug, plant_vase, shelf, wall_art), 190 images each, 1485 images in total (35 fabric-swatch listings were removed with `src/clean.py`). Seed 0.
* **Retrieval.** Leave-one-out image query, relevance = same category, NDCG@10 for HSV, HOG, LBP, CLIP. Extra row: CLIP text query
  "a photo of a <category>".
* **Segmentation.** IoU between the classical mask and the rembg mask over all images. rembg is *pseudo* ground truth.
* **Scene generation.** 13 prompts, each rendered with harmony OFF and ON. Reported: compatibility score, hue spread (circular std. of all palette hues),
  mean CLIP relevance of the chosen items, seconds per scene on CPU.
* **Hardware.** Offline run on Colab/Kaggle T4; scene timing on CPU Windows 11 laptop CPU.

## 5. Results

Figures are produced automatically in `results/figures/`:

1. `fig1_pipeline_stages.png` raw -> enhanced -> segmented -> cut-out
2. `fig2_masks_iou.png` classical vs rembg masks with per-image IoU
3. `fig3_ndcg.png` NDCG@10 bar chart
4. `fig4_harmony_on_off.png` five prompts, harmony OFF vs ON
5. `fig5_palettes.png` cut-outs with their k-means swatches

| Result | Value |
|---|---|
| NDCG@10 HSV / HOG / LBP / CLIP | 0.341 / 0.675 / 0.696 / 0.856 (`results/ndcg_table.csv`). Weakest categories: shelf (CLIP 0.70) and table (0.78) |
| NDCG@10 CLIP text query | 0.903 mean; 1.0 for 6 of 8 categories, plant_vase 0.78, wall_art 0.45 (the phrase "wall art" is a weak CLIP query) |
| Mean IoU classical vs rembg | 0.672 over 1485 items. Per category: sofa 0.87, chair 0.79, plant_vase 0.76, table 0.74, shelf 0.69, lamp 0.68, wall_art 0.56, rug 0.32 (`results/iou.json`) |
| GrabCut/full-frame fallback fraction | 11.6 % |
| Compatibility score harmony OFF vs ON | 0.914 vs 0.950 (13 prompts; ON reaches 0.95 in 13/13, OFF is lower in 7/13) |
| Hue spread OFF vs ON; CLIP relevance OFF vs ON | 17.0 deg vs 10.8 deg; 0.3071 vs 0.3065 (harmony costs almost no text relevance) |
| Median seconds per scene (CPU) | 0.51 s (0.48 OFF, 0.68 ON) on the dev laptop |

**How to read them (write after seeing the numbers).** Expect CLIP to beat the hand-crafted features on category retrieval because
HSV/HOG/LBP see colour, edges or texture but not object identity; check whether this holds. State clearly that NDCG is category-based, so the harmony
re-ranking does not change it by design. Harmony is evaluated with the palette score, and because harmony-ON optimises that same score the gain is partly by
construction; use hue spread and the CLIP relevance drop as a second view. The score saturates at 0.95 (the neutral-pair value), so prompts whose OFF choice already has neutral colours show no change.

## 6. Failure cases (examples to find in the figures and describe)

* **White object on white background** (lamp shades, pale chairs): the Otsu difference image is near zero, so the classical mask loses the object. GrabCut sometimes recovers it.
* **Lifestyle photos instead of product shots** (rugs and sofas shown inside a room): the "foreground" is the whole room; both methods give plausible-looking but different masks, so IoU is low (about 0.4 for rugs in the smoke test).
* **rembg errors**: U2-Net can keep only part of an object (a lamp head without its stand) or merge neighbouring objects, so IoU against it is not ground truth.
* **Gray-world on single-colour products**: beige chairs turned blue before we reduced the correction to half strength.
* **Low resolution**: upscaled ~100 px images look soft in the scene.
* **Composition**: no real perspective or lighting; cut-outs may show matting halos and leftover pieces of other objects; items that rest on furniture are placed by fixed coordinates.
* **Retrieval**: ABO titles for "chair" include bar stools and dining sets, so the "chair" category is heterogeneous and CLIP "armchair" queries may return something else.
* **Harmony**: neutrals are treated as compatible with everything, and the score ignores lightness, pattern and style.

## 7. Future work

* **3D export.** ABO provides GLB models for many items. Retrieve the matching GLB, place it with the same slot rules, and export a combined
  scene as GLB (e.g. with `trimesh`), viewable in a browser with `<model-viewer>` or Gradio's `Model3D`. Cut-out sizes and floor/wall rules map directly onto 3D transforms.
* Real ground-truth masks (hand-label ~50 images) to replace the pseudo ground truth.
* Better placement: homography-based floor perspective, estimated light direction for shadows, depth ordering.
* Learned harmony or aesthetic scoring, user studies with humans comparing harmony ON/OFF.
* Larger catalogue, price and style filters, higher resolution images (ABO `original` images).

## References

See the Credits section of `README.md`.

## Appendix: data issues found

* 35 ABO listings were fabric swatches ("Khaki Swatch, ...") filed under chair/sofa; removed.
* 250 of 1485 items have classical/rembg IoU < 0.3 and are excluded from scenes (mostly rugs shown in room photos and wire shelves).
* 23 items initially had empty masks from both methods (swatches and thin wire shelves); a full-frame fallback was added.
