# ShelfScanYOLO — Validated Demo Images

Every image below was scored with **measured ONNX output** (not eyeballed). The exact
inference + scoring pipeline is in `../validate_demo.py`; re-run with:

```bash
python validate_demo.py --score demo_images/demo_gt.json
```

Model: `shelfscan-yolo11s-sku110k.onnx` (YOLO11s, trained on SKU-110K, single class `object`).
Pipeline: letterbox→640, /255, RGB, NCHW → ONNX → decode channel-major `[1,5,8400]` →
threshold conf 0.25 → cxcywh→xyxy → **invert letterbox to original px** → NMS IoU 0.45.
Score is used **as-is** (already sigmoid'd in-graph — no re-sigmoid). Recall/precision are
measured at **IoU ≥ 0.50** via greedy 1-1 pred→GT matching, highest-confidence first.

Pipeline correctness was verified on raw output: box channels max ≈ 639.9 (confirms 640
letterbox pixel space) and score channel range 0.0–0.83 (confirms sigmoid already applied).

---

## Selected demo images (3)

| File | Resolution | GT products | Detected | Recall @IoU0.5 | Precision @IoU0.5 | Mean conf |
|------|-----------|-------------|----------|----------------|-------------------|-----------|
| `shelf_ultradense_499.jpg` | 3120×4160 | 499 | 491 | **0.870** | 0.884 | 0.664 |
| `shelf_dense_216.jpg`      | 3120×4208 | 216 | 221 | **0.972** | 0.950 | 0.766 |
| `shelf_clean_155.jpg`      | 2592×1936 | 155 | 159 | **1.000** | 0.975 | 0.778 |

Overlay for each: `*_overlay.png` — **green** = predicted box (+ headline count), **red** =
ground-truth box. Green boxes wrap individual product facings and align tightly over the red GT.

### `shelf_ultradense_499.jpg`  — the money shot
- **Source:** SKU-110K test split, via HF `harryrobert/SKU-110k-reformat` (mirror of
  `eg4000/SKU110K_CVPR19`, Goldman et al., CVPR 2019). Record index 53.
- **GT / detected / recall / precision / conf:** 499 / 491 / 0.870 / 0.884 / 0.664
- **Why this photo:** a wall of ~500 packed boxes — the single most impressive trade-show
  frame for the "hundreds of boxes drawn on-device, no upload" story. Recall 0.87 on 499
  facings is a genuinely hard, honest number (not a cherry-picked easy frame).

### `shelf_dense_216.jpg`  — dense + near-clean
- **Source:** SKU-110K test split, HF `harryrobert/SKU-110k-reformat`. Record index 96.
- **GT / detected / recall / precision / conf:** 216 / 221 / 0.972 / 0.950 / 0.766
- **Why this photo:** high-density shelf where the model still nails 97% of facings at 95%
  precision — the "dense *and* accurate" proof point.

### `shelf_clean_155.jpg`  — clean high-confidence shelf
- **Source:** SKU-110K test split, HF `harryrobert/SKU-110k-reformat`. Record index 119.
- **GT / detected / recall / precision / conf:** 155 / 159 / 1.000 / 0.975 / 0.778
- **Why this photo:** an evenly-lit shampoo/bottle shelf with **100% recall** and 0.98
  precision — the crisp, unambiguous frame for showing tight, clean boxes.

---

## Aggregate model accuracy (honest, full tested set)

Measured over **36 SKU-110K test shelves** spanning easy→ultra-dense (GT counts 28→499),
sampled across the density spectrum. All numbers @IoU≥0.5:

- **Recall:** median **0.903**, mean 0.847 (min 0.179, max 1.000)
- **Precision:** median **0.894**, mean 0.834 (min 0.135, max 0.986)
- **31/36 shelves (86%) clear the recall ≥ 0.70 target**; 24/36 clear recall ≥ 0.85.
- On the **dense** subset (GT ≥ 100 facings, 34 shelves) — the real target regime —
  **median recall 0.908**.

### Caveats (stated loudly, not hidden)
- **Sparse shelves are the weak spot.** The 2 sparse images (GT < 100) had median recall
  0.43; the single worst case was a 28-product shelf at recall 0.18. The model is tuned for
  *densely packed* scenes and under-detects on sparse/atypical layouts.
- **Two dense outliers underperformed** (recall 0.40 and 0.54) on odd lighting / reflective
  packaging — real failure modes, not selection artifacts.
- **conf 0.25 / NMS IoU 0.45** are the spec defaults and were not tuned per-image; a live app
  may raise conf for cleaner boxes or lower it to chase recall on very dense walls.
- Metrics are single-class localization only (SKU-110K has one class `object`); this measures
  *product-facing detection*, not brand/SKU classification.

## Licensing — FLAG
All images are from **SKU-110K** (Goldman et al., CVPR 2019). SKU-110K is distributed for
**academic / research use, non-commercial** (the mirror `PrashantDixit0/SKU-110K` tags it
`cc-by-nc-2.0`). These frames are fine for an internal/research **benchmark demo**, but are
**NOT cleared for commercial redistribution**. If ShelfScanYOLO is productized or shown in a
commercial context, replace these with owned/permissively-licensed shelf photos.
