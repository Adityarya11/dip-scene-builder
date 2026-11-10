# Handout: AI-Assisted 3D Commerce Scene Builder

Read this once from top to bottom and you will know what we built, how each part works, which course topics it covers, what the numbers mean, and how to answer the likely questions. Every number below comes from the files in `results/`. Code references are `file.function`.

---

## 1. What we built, in plain words

Online shops show one product at a time on a white background. A buyer cannot see how a sofa, a lamp and a rug look together. We built a system where you type a short idea, for example **“cozy reading corner”**, and it:

1. finds matching product photos from the Amazon Berkeley Objects (ABO) dataset,
2. cuts each product out of its background,
3. picks a set of products whose colours go together,
4. places them in a room and returns a **2D room picture**,
5. exports the same choice as a **3D room file (GLB)** that can be rotated in the browser.

It runs as a Gradio app (`app.py`) on a free CPU Hugging Face Space.

**Honest framing.** Most of the pipeline is classical image processing that we wrote (filters, histogram equalisation, colour spaces, thresholding, morphology, clustering, compositing). Two deep models are used **pre-trained and unchanged**: CLIP (text and image matching) and U2-Net (background removal). We trained nothing. The 3D room is made of simple boxes, cylinders and spheres coloured from the chosen products. It is **not** a 3D reconstruction of the photos.

**Data.** 1,485 products in 8 categories: lamp, plant_vase, rug, shelf, table, wall_art (190 each), chair (174), sofa (171). Only ABO’s small images (about 100 to 256 px) and listing titles; no 3D models. The full set of cut-outs and vectors is 30 MB.

---

## 2. Architecture at a glance

**Offline (run once on Colab, T4 GPU, about 25 minutes)** produces `artifacts/`:

```
ABO listings + small photos  ->  enhance  ->  segment (2 ways)  ->  cut-outs (RGBA webp)
                                    |                                   |
                                    +--> features (HSV, HOG, LBP, CLIP) +--> k-means palettes
                                                       |
                                         artifacts/ (embeddings.npy, palettes.npz, meta.csv, cutouts/)
```

**Online (every prompt, about 0.5 to 0.6 s on a laptop CPU):**

```
prompt -> scene template (slots) -> CLIP text vector -> top-5 per slot
       -> best of 5^n sets (CLIP score + 3 x colour harmony)
       -> placement rules + Reinhard colour match + Gaussian shadows + alpha blend  -> 2D scene
       -> export_scene3d (trimesh)                                                   -> 3D GLB
```

The app loads only the precomputed vectors, palettes and cut-outs, plus the CLIP **text** encoder. Startup was about 10 s on a laptop.

### File map

| File | Job |
|---|---|
| `src/data.py` | Download ABO listings, `images.csv`, small images; build `meta.csv`. Resumable; seed 0; skips swatch listings |
| `src/preprocess.py` | resize, denoise, gray-world white balance, CLAHE |
| `src/segment.py` | classical mask (Otsu + morphology, GrabCut fallback), U2-Net mask via rembg, IoU, cut-out |
| `src/features.py` | HSV histogram, HOG, LBP, CLIP image and text embeddings |
| `src/index.py` | cosine-similarity search, same class for every feature type |
| `src/palette.py` | k-means palettes in LAB, colour-harmony score, hue spread |
| `src/compose.py` | prompt to slots, choose items, placement, colour match, shadows, blending, room |
| `src/scene3d.py` | 3D room from proxy shapes, exported as GLB (written by our teammate) |
| `src/build.py` | runs enhance, segment, cut-out, features for all images |
| `src/clean.py` | removes fabric-swatch listings from an existing `artifacts/` folder |
| `src/eval.py` | all metrics and figures |
| `app.py` | Gradio app: prompt, harmony checkbox, 2D scene, gallery with palettes, 3D viewer |
| `notebooks/run_all.ipynb` | Colab/Kaggle runner for the offline build |

---

## 3. The pipeline stage by stage

### 3.1 Data (`src/data.py`)
ABO listing files give product type, English title and image id; `images.csv` gives the path of each small image. We map product types to 8 categories, keep listings with an English title, shuffle with a fixed seed and take up to 190 per category. Swatch listings (titles such as “Khaki Swatch, Stone & Beam”) are skipped, because they are fabric samples, not products.

### 3.2 Enhancement (`src/preprocess.py`), order: resize, denoise, white balance, contrast
- **Resize** so the longest side is 256 px. Bicubic when enlarging (most images), area averaging when shrinking.
- **Denoise**: 3×3 median filter (removes JPEG block noise) then Gaussian blur, sigma 0.8.
- **Gray-world white balance**: scale each channel so the average colour of the *object* becomes grey. We compute the average only on pixels that are not almost white (minimum channel below 235), because the white background would dominate it. We apply **half** of the correction and clip gains to 0.85 to 1.18. Reason: our first version turned beige chairs blue, since a beige object really is beige.
- **CLAHE** (contrast-limited adaptive histogram equalisation), clip limit 2.0, 8×8 tiles, on the **L channel of LAB only**, so contrast improves and colours do not change.

### 3.3 Segmentation (`src/segment.py`)
**Classical:**
1. Estimate the background colour as the median of the one-pixel frame of the image.
2. Distance image = largest absolute channel difference from that colour; blur 3×3.
3. **Otsu threshold** on the distance image.
4. **Morphology**: opening and closing with a 5×5 ellipse, keep the largest connected component, fill interior holes.
5. If the mask covers under 8 % or over 90 % of the image it is implausible, so run **GrabCut** (graph-cut segmentation) with a rectangle just inside the border. If GrabCut also fails (under 2 %), use the whole frame.

**U2-Net via rembg:** a deep salient-object network. It is our **reference**, not ground truth (ABO small images have no human masks).

**Which mask becomes the cut-out:** the U2-Net mask; the classical mask if U2-Net returns almost nothing; for **wall art** the bounding rectangle of the classical mask, because U2-Net cuts out the motif inside a frame (a dog, a horse) instead of the frame. Cut-outs are RGBA webp, 256 px, with a 1 px soft alpha edge.

**Agreement and quality filter:** IoU between the two masks is stored per item. 250 items have IoU below 0.3 (109 rugs, 51 wall art, 32 shelves, the rest 58 spread over other categories). Wall art is exempt because it uses a rectangle mask, so the other **199 items are never used in scenes**; more than half of the rugs are among them, because many rug photos show a whole room.

### 3.4 Features (`src/features.py`) and search (`src/index.py`)
All vectors are L2-normalised, so a dot product is the cosine similarity.
- **HSV histogram**: 8 hue × 4 saturation × 4 value bins = 128 numbers, computed on the object only (colour).
- **HOG**: 128×128 grey image, 9 orientations, 16 px cells (edges and shape).
- **LBP**: uniform pattern, 8 neighbours, radius 1, histogram over a 4×4 grid = 160 numbers (texture).
- **CLIP ViT-B/32** (OpenCLIP, OpenAI weights): 512 numbers for an image or a text; text and image live in one space.

`Index.search` returns the top-k by cosine similarity, with optional category mask and leave-one-out. The same class serves all four feature types.

### 3.5 Palettes and the harmony score (`src/palette.py`)
- **Palette:** k-means on the object’s pixels in **CIELAB** with k in {3, 4, 5} chosen by silhouette score. LAB is used because distances there approximate perceived colour differences.
- **Item summary:** circular mean hue and chroma of the chromatic clusters. An item is **neutral** if its chroma is below 18 or less than 15 % of it is coloured.
- **Pair score** (0 to 1) = 0.7 × hue + 0.3 × chroma agreement. Hue score: 1 for analogous (close hues), 0.9 at complementary (180° apart), 0.7 at triadic (120° apart), falling off linearly. A pair with a neutral item scores 0.95.
- **Scene harmony** = mean pair score. Because neutrals score 0.95, the score saturates near 0.95.
- **Hue spread**: circular standard deviation in degrees of all chromatic palette colours in a scene. Lower means colours sit in a narrow band. We report it as a second view.

### 3.6 Composition (`src/compose.py`)
1. **Prompt to slots.** Keyword aliases choose one of six templates: reading corner, living room, home office, dining room, bedroom, entryway. Each template lists slots (category, phrase, x, y), for example reading corner = bookshelf, rug, armchair, floor lamp, plant in pot. Remaining words (cozy, modern, rustic) are style words added to the CLIP text, such as “a photo of a cozy armchair”.
2. **Retrieve** the top 5 per slot by CLIP cosine similarity inside the slot’s category, excluding bad cut-outs.
3. **Choose the set.** With 5 candidates in 5 slots there are 3,125 combinations. Score = mean normalised CLIP score + 3 × harmony, where each CLIP score is (similarity minus the slot’s best) / 0.05. **Harmony off** takes the single best match per slot.
4. **Placement rules.** Per-category size priors (for example sofa 40 % of canvas width, lamp 50 % of height), floor items sit on a floor line, wall art is centred on the wall, rugs are squashed to 35 % height to look flat on the floor, chairs on the right are mirrored to face the centre, farther items are slightly smaller.
5. **Reinhard colour transfer** at 25 % strength: match the object’s LAB mean and standard deviation toward the room’s, then blend with the original. Full transfer would repaint the product.
6. **Shadows**: an ellipse as wide as the object’s feet, blurred with a Gaussian (sigma 5). Wall items get an offset blurred silhouette.
7. **Alpha blending** onto a **procedural room**: gradient wall tinted slightly with the scene’s average hue, wood floor with converging plank lines, baseboard line, film grain, vignette.

### 3.7 3D export (`src/scene3d.py`)
Takes the same selected items and builds a room of simple solids: floor, back wall, left wall, baseboards, then a proxy shape per category (sofa and chair = box legs, seat, back, arms; table = top and four legs; lamp = cylinders and a sphere bulb; shelf = frame, shelves, a few books; plant = pot, trunk, green spheres; rug = flat boxes with stripes; wall art = frame). Colours come from the chosen product’s palette: the largest vivid non-white swatch as primary and a clearly different swatch as secondary. Positions are fixed per category and do **not** copy the 2D layout. The scene is rotated from Z-up to the Y-up convention of glTF and exported as GLB with trimesh. Export takes about 0.05 s, the file about 50 KB. The app shows it with Gradio’s 3D model viewer.

---

## 4. How the course maps to the project

Source: *Digital Image Processing 3-0-2, Course Structure 2026-27*. Primary text: Gonzalez and Woods, *Digital Image Processing*, 4th ed. Companion: Szeliski, *Computer Vision*.

| Module | Syllabus topics | What we did (and where) | Coverage |
|---|---|---|---|
| **M1** Introduction (Gonzalez Ch. 1–2) | pixels, resolution, sampling, quantisation, representation, DIP vs CV | 100–256 px source photos; bicubic and area resampling (`preprocess.resize_long_side`); 8-bit RGB arrays and RGBA cut-outs (`segment.cutout`); enhancement and segmentation are DIP, CLIP retrieval is CV | Direct |
| **M2** Spatial enhancement (Ch. 3) | intensity transforms, histogram equalisation, contrast stretching, log/gamma, image arithmetic | CLAHE on L (`preprocess.clahe_l`); gray-world gains with clipping (`preprocess.gray_world`); alpha blending (`compose.blend`). Not used: log and gamma | Direct |
| **M3** Spatial filtering (Ch. 3) | convolution, smoothing, sharpening, noise, median, Gaussian, Laplacian, high-boost | median and Gaussian denoising (`preprocess.denoise`); Gaussian-blurred shadows (`compose.drop_shadow`); gradient filters inside HOG. Not used: sharpening, Laplacian | Direct |
| **M4** Frequency domain (Ch. 4) | Fourier, DFT, FFT, ideal/Butterworth/Gaussian filters, wavelets | **Not used.** Link: a Gaussian blur is a Gaussian low-pass filter, so our denoising is low-pass filtering in the spatial domain. FFT analysis is future work | **Gap** |
| **M5** Colour and morphology (Ch. 7 colour, Ch. 9 morphology) | RGB, HSV, Lab, pseudocolour, dilation, erosion, opening, closing, boundary extraction | HSV histogram (`features.hsv_hist`); LAB k-means (`palette`); Reinhard transfer (`compose.reinhard`); opening, closing, largest component, hole fill (`segment._clean`). Not used: pseudocolour, boundary extraction | Direct |
| **M6** Features (Ch. 10–11, Szeliski) | Sobel/Prewitt/Canny, Harris, texture, shape, SIFT/SURF/ORB, matching | HSV histogram, HOG (edges and shape), LBP (texture), cosine matching (`features`, `index`). Not used: Canny, Harris, SIFT, ORB | Partial |
| **M7** Segmentation (Ch. 10–11) | thresholding, Otsu, region growing, watershed, k-means segmentation, classification | Otsu (`segment.otsu_mask`); GrabCut graph cut (`segment.grabcut_mask`); k-means of pixels in LAB (`palette.dominant_colors`). Not used: region growing, watershed | Direct |
| **M8** ML for images | supervised/unsupervised, KNN, decision trees, SVM, random forest, naive Bayes, clustering | k-means with silhouette score (`palette`); nearest-neighbour retrieval (`index.search`). No trained classifier | Partial |
| **M9** Deep learning | CNN, transfer learning, detection, semantic segmentation (U-Net), captioning | U2-Net encoder–decoder segmentation via rembg (`segment.rembg_mask`), used pre-trained | Direct (pre-trained) |
| **M10** ViT and multimodal | ViT, DETR, SAM, foundation models, multimodal vision, video | OpenCLIP ViT-B/32 image and text embeddings (`features.load_clip`, `clip_images`, `clip_text`) drive all retrieval. No video | Direct (pre-trained) |

**Labs.** The syllabus labs are MATLAB (1–10) and Python (11–12). This project is Python with OpenCV, scikit-image and scikit-learn, which the syllabus lists as references. Lab 2 (histograms) = CLAHE; lab 3 (filtering) = median/Gaussian; lab 6 (colour) = HSV/LAB/Reinhard; lab 7 (morphology) = mask cleanup; lab 8 (features) = HSV/HOG/LBP; lab 9 (segmentation) = Otsu, GrabCut; lab 10 (classical ML) = k-means, nearest neighbours; lab 11 (CNN) = U2-Net; lab 12 (ViT) = CLIP. Labs 1, 4 (noise and restoration) and 5 (FFT) are not part of the pipeline.

**Chapter numbers.** The syllabus module table prints “Ch. 6 & Ch. 9” for module 5, but its textbook list gives Colour Processing as Ch. 7 and Wavelets as Ch. 6. We cite Ch. 7 for colour and Ch. 9 for morphology. The syllabus also lists graph cuts (GrabCut) among the new 4th-edition topics.

---

## 5. Results

### 5.1 Retrieval: NDCG@10
Each product image is a query; its 10 nearest products are ranked; a result is relevant if it has the **same category**. NDCG@10 = DCG / ideal DCG, where DCG = Σ rel_i / log2(i + 1). 1.0 is perfect.

| Feature | Mean | chair | lamp | plant_vase | rug | shelf | sofa | table | wall_art |
|---|---|---|---|---|---|---|---|---|---|
| HSV histogram | 0.341 | 0.234 | 0.299 | 0.401 | 0.427 | 0.283 | 0.400 | 0.323 | 0.355 |
| HOG | 0.675 | 0.629 | 0.751 | 0.713 | 0.889 | 0.448 | 0.850 | 0.377 | 0.759 |
| LBP | 0.696 | 0.690 | 0.794 | 0.713 | 0.884 | 0.437 | 0.779 | 0.440 | 0.838 |
| **CLIP** | **0.856** | 0.794 | 0.906 | 0.912 | 0.942 | 0.704 | 0.887 | 0.781 | 0.919 |

CLIP text query (“a photo of a <category>”): mean **0.903**. Six categories score 1.0, plant_vase 0.780, wall_art 0.445 (the phrase “wall art” is vague).

**Reading it.** CLIP wins because it encodes what the object is. HSV, HOG and LBP see only colour, edges or texture. Shelf and table are hard for every method because those categories mix very different products. **Caveat:** the metric is category-based, so it measures category retrieval, not style, price or colour match.

### 5.2 Segmentation: IoU, classical vs U2-Net
Mean **0.672** over 1,485 items. Per category: sofa 0.870, chair 0.794, plant_vase 0.757, table 0.737, shelf 0.686, lamp 0.679, wall_art 0.564, **rug 0.317**. 11.6 % of images needed the GrabCut or full-frame fallback. **U2-Net is pseudo ground truth**, so this measures agreement with a strong model, not true accuracy. Rugs agree worst because many rug photos show a whole room.

### 5.3 Colour harmony off vs on (13 prompts)

| | Compatibility | Hue spread | CLIP relevance | Seconds per scene |
|---|---|---|---|---|
| Harmony off | 0.914 | 17.0° | 0.3071 | 0.56 |
| Harmony on | 0.950 | 10.8° | 0.3065 | 0.77 |

Median scene time over all runs: 0.58 s (an earlier run gave 0.51 s; it varies with machine load). Harmony changed the score on **6 of 13 prompts** (boho living room, colourful reading nook, cozy reading corner, modern living room, rustic dining room, scandinavian living room); on the other 7 the best matches already scored about 0.95. **Caveat:** harmony-on maximises the same score we report, so part of the gain is by construction. That is why hue spread and CLIP relevance are shown too. NDCG does not move with harmony by design, because harmony only re-ranks inside a category.

### 5.4 Figures (in `results/figures/`)
`fig1_pipeline_stages.png` raw to cut-out · `fig2_masks_iou.png` classical vs U2-Net masks · `fig3_ndcg.png` NDCG chart · `fig4_harmony_on_off.png` five prompts, off vs on · `fig5_palettes.png` cut-outs with palette swatches.

---

## 6. Problems we found and fixed (good to mention)

| Problem | Fix |
|---|---|
| Gray-world turned beige chairs blue | foreground-only statistics, half-strength, clipped gains |
| U2-Net shattered framed pictures into fragments | wall art uses the classical mask’s bounding rectangle |
| 35 fabric swatches were counted as chairs and sofas | removed (`src/clean.py`, and `data.py` now skips them) |
| 23 items had empty masks (swatches and thin wire shelves) | full-frame fallback in `classical_mask` and `build` |
| Bad cut-outs appeared in scenes | items with IoU below 0.3 are excluded from composition |
| Merge of the 3D branch brought back the un-cleaned artifacts | re-ran `clean`, `palette`, `eval`; numbers in this handout are from the cleaned set |
| Scene evaluation was missing from the first Colab output | re-ran evaluation locally after palettes existed |

---

## 7. Honest limits

- ABO small images are low resolution, so cut-outs are soft.
- U2-Net is a reference, not human ground truth.
- NDCG is category-based.
- Harmony evaluation is partly circular (see 5.3).
- Many products are neutral colours, which the harmony rule treats as compatible with everything, so differences are small; the score saturates near 0.95.
- 199 of 1,485 items are excluded from scenes because the two segmenters disagree (109 are rugs). Rug cut-outs that remain often keep a strip of floor.
- The 2D scene has a fixed camera and no real perspective or lighting.
- **The 3D room is proxy geometry. It is not reconstructed from the photographs.** Every sofa has the same shape; only the colours come from the product.
- The frequency-domain module (M4) is not covered.

---

## 8. Questions you may be asked

**What is new here?** Nothing in a single block is new. The contribution is the complete path from a sentence to a room, and the harmony-aware choice of products. We say that plainly.

**Why not train your own model?** The course goal is image processing. CLIP and U2-Net are pre-trained and good enough for retrieval and segmentation; our work is in the classical stages around them and in their evaluation.

**Why CLIP and not HOG or colour?** Measured: NDCG@10 0.856 against 0.675 and 0.341 (5.1). CLIP encodes object identity; the others do not.

**What is NDCG?** A ranking score that rewards putting relevant items near the top, normalised so 1.0 is perfect (5.1).

**What is IoU and why is it pseudo ground truth?** IoU is overlap divided by union of two masks. We have no human masks, so we compare against U2-Net. It shows agreement, not accuracy.

**Why Otsu on a distance image?** Product photos have a uniform background. The distance from the background colour splits object and background into two clear groups, which is the situation Otsu assumes.

**What is GrabCut?** A graph-cut method that models foreground and background colours with Gaussian mixtures and finds a minimum cut. We start it with a rectangle just inside the border.

**Why LAB?** Euclidean distance in LAB is close to perceived colour difference, so k-means groups colours the way people see them. LAB also separates lightness from colour, which CLAHE and Reinhard transfer use.

**Why half-strength gray-world?** Full correction repaints single-colour products. Half keeps most of the product’s true colour (6).

**How is k chosen for k-means?** Silhouette score over k = 3, 4, 5.

**What is Reinhard colour transfer?** Shift and scale the LAB channels of an object so its mean and standard deviation approach those of a target. We use the room as the target at 25 % strength.

**Is the harmony result real or circular?** Partly circular, because the choice maximises the reported score. Hue spread and CLIP relevance are an independent-ish check: hue spread falls from 17° to 11° and relevance barely changes. A user study would settle it.

**Is the 3D model real?** No. It is proxy geometry coloured from the products. The photos are not reconstructed.

**Where is the frequency domain?** Not used. Gaussian blur is a low-pass filter; an FFT analysis of the photos is what we would add.

**Why only small images and 8 categories?** Free CPU hosting and a one-day budget: the whole project is 30 MB. The cost is soft cut-outs.

**How does the app start so fast?** All heavy work is precomputed. The app loads vectors, palettes and cut-outs and only the CLIP text encoder.

**How do you make sure it is repeatable?** Fixed seed, resumable downloads and builds, and scripts for every step.

**What would you do with more time?** Use ABO’s own 3D models for the chosen items, hand-label about 50 masks, add FFT analysis, run a user study on harmony.

---

## 9. How to run it

**Local smoke test (CPU, few minutes):**
```bash
pip install -r requirements-build.txt
python -m src.data  --per-cat 6 --out data_test
REMBG_MODEL=u2netp python -m src.build --data data_test --out artifacts_test
python -m src.palette --art artifacts_test
python -m src.eval --data data_test --art artifacts_test
ART_DIR=artifacts_test python app.py
```
**Full build:** open `notebooks/run_all.ipynb` on Colab with a T4 GPU, set `REPO_URL`, Run all. It downloads about 1,500 images, builds `artifacts/` and `results/`, and offers `outputs.zip`. Then run `python -m src.clean --art artifacts` and `python -m src.palette --art artifacts` and `python -m src.eval --data data --art artifacts --light` locally if the swatch listings are still in the set.
**Deploy:** create a Gradio Space, copy `app.py`, `src/`, `requirements.txt`, `README.md` (its header has `sdk: gradio`) and `artifacts/` (without `artifacts/masks/`), then push. See `README.md`.

---

## 10. Glossary

- **CLAHE**: contrast-limited adaptive histogram equalisation; equalises contrast per tile with a clip limit.
- **CIELAB (LAB)**: colour space with lightness L and two colour axes a, b, close to human perception.
- **Gray-world**: assumes the average scene colour is grey and scales channels to match.
- **Otsu**: picks the threshold that best separates two groups of pixel values.
- **Morphology**: opening (erode then dilate) removes specks; closing (dilate then erode) fills gaps.
- **GrabCut**: graph-cut foreground extraction started from a rectangle.
- **U2-Net**: nested U-shaped network for salient-object (background) segmentation.
- **HOG / LBP**: gradient-orientation histogram (shape) / local binary pattern histogram (texture).
- **CLIP**: model that maps images and text into one vector space.
- **Cosine similarity**: dot product of unit vectors; 1 means the same direction.
- **NDCG@10**: ranking quality of the top 10 results, 1.0 is perfect.
- **IoU**: intersection over union of two masks.
- **k-means / silhouette**: clustering into k groups / score of how well-separated the groups are.
- **Reinhard transfer**: colour transfer by matching channel mean and standard deviation.
- **GLB**: binary glTF file, a compact 3D scene format.

---

## 11. Credits

Amazon Berkeley Objects (Collins et al., CVPR 2022, CC BY-NC 4.0) · OpenCLIP and CLIP (Radford et al., 2021) · rembg and U2-Net (Qin et al., 2020) · GrabCut (Rother et al., 2004) · CLAHE (Zuiderveld, 1994) · Reinhard colour transfer (2001) · HOG (Dalal and Triggs, 2005) · LBP (Ojala et al., 2002) · OpenCV, scikit-learn, scikit-image, NumPy, pandas, trimesh, Gradio. Course references: Gonzalez and Woods, *Digital Image Processing*, 4th ed.; Szeliski, *Computer Vision: Algorithms and Applications*. Data licence is non-commercial; this is a course project.
