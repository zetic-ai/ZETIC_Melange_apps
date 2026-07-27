# Model selection — VisualSearch (Vision / on-device visual search, use-case: snap a product → localize → embed → catalog match; fashion e-commerce "Musinsa wedge")

Two-model pipeline: **(1)** a lightweight object detector localizes the salient
product/object; **(2)** a mobile image-embedding tower turns the crop into a compact
512-d feature vector. Only the vector would leave the device. Both slots were chosen by
**VALIDATION-GATED SELECTION** (EXPLORATION.md §7): head-to-head measurement on one
shared, real, labeled image set — not shortlist reasoning.

## Shared validation set (real e-commerce product photos)
- Source: `Marqo/deepfashion-inshop` (In-Shop Clothes Retrieval Benchmark), 224×224.
- 60 images across **16 distinct product instances** and **14 apparel categories**
  (denim, pants, shirts, shorts, sweatshirts, tees, blouses, cardigans, dresses,
  graphic tees, jackets, rompers, skirts, sweaters), men + women.
- Each product instance has **3–4 views** (front / side / back / flat) — this gives a
  true **instance-level** retrieval signal (same physical product, different photo), the
  real visual-search task, plus a coarser **category-level** signal.
- Reproducible: `scratchpad/build_valset.py` (row-group sampling + manifest).

---

## SLOT 1 — DETECTOR

### Shortlist (top 5)
| Rank | HF repo | DL / Likes | License | Export path | Melange-fit notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1 | **Ultralytics YOLO11n (COCO)** | n/a (Ultralytics assets) | AGPL-3.0 | Ultralytics ONNX (repo-proven recipe) | nano 2.6M params, 10.7 MB ONNX, static [1,3,640,640]→[1,84,8400], all-standard ops, 80 COCO classes covers the broad "product/object" story (bottle, backpack, handbag, cell phone, laptop, cup…). Same recipe as PyroGuard/ShelfScan/DentalXray. | **9.2** |
| 2 | kesimeg/yolov8n-clothing-detection | 20 likes | AGPL-3.0 | Ultralytics ONNX | nano 3.0M, fashion-specific (accessories/bags/clothing/shoes), boxes garments directly — but 4 classes only, fashion-ONLY, fragments a garment into multiple boxes. | 7.6 |
| 3 | yainage90/fashion-object-detection | 3.4k DL / 38 likes | MIT | transformers (Conditional-DETR) | best fashion signals, MIT — but ResNet-50 DETR ≈ 43M params (~170 MB): **too big for mobile**, + DETR attention/Hungarian export friction. | 5.0 |
| 4 | itesl/yolos-tiny-clothing | 3 DL | Apache-2.0 | transformers (YOLOS/ViT-DETR) | smaller ViT-DETR, permissive — but weak signals, transformer-DETR export risk, set-prediction head. | 4.2 |
| 5 | kiadamson/detr-fashion-detection | 1 DL | — | transformers (DETR) | DETR family, negligible adoption, heavy. | 3.5 |

### Measured head-to-head (60 real product photos, conf 0.25, onnxruntime/Ultralytics)
"Sensible primary box" = top-confidence box covering ≥8% area and reasonably central (center-offset <0.30).

| Model | detection rate | sensible-primary hit | mean primary area-frac | mean center-offset | mean #boxes | primary-label dist |
|-------|---------------|----------------------|------------------------|--------------------|-------------|--------------------|
| **YOLO11n (COCO)** | 96.7% | **95.0%** | 0.39 | **0.04** | 1.17 | person 55, clock 1, vase 1, fire-hydrant 1 |
| kesimeg clothing | 100% | 81.7% | 0.21 | 0.12 | 2.77 | clothing 59, shoes 1 |

### Winner: **Ultralytics YOLO11n (COCO)**
Why this one over the runners-up (Melange-fit + task-fit trade-off):
- **Cleaner salient-object localization:** one large, well-centered primary box 95% of
  the time (area 0.39, offset 0.04, ~1 box/img). The clothing model detects garments
  directly but *fragments* them (top+bottom → 2.77 boxes/img) and its primary box is
  smaller and more off-center (81.7% hit) — worse for "crop the one salient object".
- **Generality matches the stated use-case.** The demo is "snap ANY product/object"
  (gadgets, bottles, bags — explicitly). YOLO11n's 80 COCO classes cover those; the
  clothing model (4 classes) cannot localize a bottle/gadget at all.
- **Lowest Melange risk:** identical export recipe and op-set already proven on
  PyroGuard, ShelfScan, DentalXray, SafetyPPE in this repo.
- **Honest caveat (localizes via "person"):** on on-model fashion shots YOLO11n's top
  box is usually the *person* wearing the garment (55/60), not a garment-class box.
  That is *fine for this pipeline* — the person crop contains the outfit, and the
  embedding operates on the crop. On flatlay shots with no COCO object (~3–5%), no box
  clears threshold; the app **falls back to a center-crop / whole-image embed**.
- **License flag (AGPL-3.0):** Ultralytics YOLO is AGPL-3.0 (copyleft). This is the
  **established repo precedent** for these GTM demos (6+ shipped Ultralytics YOLO apps),
  and the Melange artifact is inference-only — but AGPL must be honored/reviewed before
  any commercial *productization* beyond the demo.

---

## SLOT 2 — EMBEDDING (image tower only; text tower out of scope)

### Shortlist (top 5)
| Rank | HF repo / timm id | DL / Likes | License | Export path | Melange-fit notes | Score |
|------|-------------------|-----------|---------|-------------|-------------------|-------|
| 1 | **MobileCLIP2-S0 image tower** — `timm:fastvit_mci0.apple_mclip2_dfndr2b` | 10k DL | apple-amlr ⚠ | torch.onnx (FastViT, standard ops) | 11.4M params, 45.8 MB ONNX, static [1,3,256,256]→[1,512], plain [0,1] input (no mean/std), mobile-designed, strong retrieval. | **9.3** |
| 2 | MobileCLIP-S0 image tower — `fastvit_mci0.apple_mclip` (v1) | 226k (family) | apple-amlr ⚠ | torch.onnx | identical size/shape; measurably weaker than v2 (see below). | 8.6 |
| 3 | DINOv2-small — `timm:vit_small_patch14_dinov2` | 315+ | **Apache-2.0** ✓ | torch.onnx (ViT, pos-embed interp) | fully-permissive, 384-d — but 22M params / **88 MB** (2× budget) and much weaker on this task (see below). | 6.4 |
| 4 | Marqo/marqo-fashionCLIP | 30k / 34 | Apache-2.0 ✓ | transformers/open_clip | fashion-tuned, permissive — but ViT-B/32 image tower (~88M params, ~350 MB): **too big for mobile**. | 5.5 |
| 5 | TinyCLIP-ViT-8M / -39M (`wkcn/...`) | 47k–287k | MIT ✓ | open_clip/onnx | MIT + small (8M tower ≈ 32 MB) — but weaker retrieval than MobileCLIP2 and less clean timm export path. | 6.0 |

(MobileCLIP2-S2 `fastvit_mci2` = 35.8M params / **143 MB** → rejected on size; text towers of all CLIP models are out of scope — image tower only.)

### Measured head-to-head (60 real photos, 256×256 /255, cosine NN, whole-image embed)
Reject degenerate models (collapsed embeddings / near-zero margin). None collapsed.

| Model | dim | size | instance top-1 | category top-1 | instance cos-margin | category cos-margin |
|-------|-----|------|----------------|----------------|---------------------|---------------------|
| **MobileCLIP2-S0** | 512 | 45.8 MB | **90.0%** | 98.3% | **0.339** | 0.302 |
| MobileCLIP-S0 (v1) | 512 | 45.8 MB | 88.3% | 98.3% | 0.274 | 0.245 |
| DINOv2-small | 384 | 88 MB | 58.3% | 70.0% | 0.225 | 0.211 |

### End-to-end through BOTH exported ONNX (real app pipeline: detect → crop → embed, onnxruntime)
- detector produced a box: **57/60 (95.0%)**
- **crop-based** retrieval: instance top-1 **85.0%**, category top-1 **95.0%**
- cosine same-instance **0.758** vs diff-instance **0.455** → margin **0.303** (healthy, non-degenerate)
- (crop-based is a touch below whole-image because cropping to the person box tightens
  context; still strong and it is the true on-device number.)

### Winner: **MobileCLIP2-S0 image tower (`fastvit_mci0.apple_mclip2_dfndr2b`)**
Why this one over the runners-up:
- **Best measured retrieval at mobile size.** 90% instance top-1 / 98.3% category and the
  largest cosine margins, at 11.4M params / 45.8 MB with dead-simple preprocessing
  (resize→/255, no mean/std) and a clean fixed 256×256 → flat [1,512] graph. Beats
  MobileCLIP-v1 across the board and crushes DINOv2-small (58%), which is also 2× the size.
- **L2-normalization baked into the ONNX graph** → cosine similarity is a plain dot
  product on-device (no Dart post-processing for the embedding).
- **⚠ LICENSE FLAG (loud):** the weights are **`apple-amlr`** (Apple Machine Learning
  Research Model license) — a **non-standard, potentially research-restricted** license.
  It is very likely fine for an internal **trade-show demo** (on-device, only vectors
  leave the device), but it is **NOT a clean permissive license** and MUST be
  legally reviewed before any commercial productization. If a fully-permissive license
  is a hard requirement, the fallback is **TinyCLIP-8M (MIT)** or **DINOv2-small
  (Apache-2.0)** — both were weaker on this task, so it is a real quality trade-off.

---

## Export
- Recipe: `export.py` (one file, both models, re-runnable). See `export.py` header.
- **Detector** — input `images` float32[1,3,640,640] NCHW RGB, letterboxed, /255 [0,1];
  output `output0` float32[1,84,8400] channel-major (cx,cy,w,h + 80 class scores, 640
  space); NMS **NOT** baked in. Opset 12, static shapes confirmed (0 dynamic axes).
  ONNX verified in onnxruntime on a real image (top score 0.923).
- **Embedding** — input `image` float32[1,3,256,256] NCHW RGB, /255 [0,1], mean 0/std 1;
  output `embedding` float32[1,512], **L2-normalized in-graph**; no external
  post-processing. Opset 14, static shapes confirmed (0 dynamic axes). ONNX verified vs
  torch: **cosine = 1.000000** on a real image; output norm |o| = 1.00000.
