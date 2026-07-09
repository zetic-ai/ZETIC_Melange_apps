# Model selection — ShelfScanYOLO (YOLO object detection, use-case: retail-shelf / warehouse SKU detection)

Target sector: retail execution & merchandising CV + warehouse CV (Infilect/InfiViz, Trax,
Shopic, Arvist). Enterprise buyers; the pitch is a free on-device auto-benchmark across
fragmented cheap Android handsets + one edge SoC, drawing SKU boxes on-device with no upload.

Core task-fit point: the flagship retail-execution task is **dense SKU / product-facing detection
on a packed shelf** (one box per product facing) — this is what Trax/Infilect do. The canonical
dataset for it is **SKU-110K** (Goldman et al., dense retail shelves). A generic COCO YOLO does
not do dense single-class shelf detection well, so training dataset is treated as a first-class
task-fit factor. Constraint from the assignment: the export recipe is the **Ultralytics YOLO**
recipe, so non-Ultralytics families (DETR / RT-DETR / D-FINE) are out of scope even when popular.

## Shortlist (top 5)

| Rank | HF repo | Downloads / Likes | License | Training data / classes | Export path | Melange-fit notes | Score /10 |
|------|---------|-------------------|---------|-------------------------|-------------|-------------------|-----------|
| 1 | **chistopat/sku110k-yolo11-object-detector** (`weights/sku110k-yolo11-s640.pt`, YOLO11s) | 179 / 0 | other (SKU-110K terms) | SKU-110K, 1 class `object` (product facing). mAP50 0.927, mAP50-95 0.577 | Ultralytics YOLO → ONNX; **already ships a fixed-shape [1,3,640,640] ONNX** | Same proven YOLO11 graph as PyroGuard; standard ops; clean opset-12 static export verified. Dense single-class SKU detection = exact Trax/Infilect task; visually the most impressive trade-show demo (hundreds of boxes on a shelf). **License flag:** research/D&D terms tied to SKU-110K. | **9** |
| 2 | foduucom/product-detection-in-shelf-yolov8 | 443 / 20 | AGPL-3.0 (ultralyticsplus base) | Retail shelves, 2 classes `Empty Shelves`, `Magical Products` | Ultralytics YOLO → ONNX (`best.pt`) | Most-liked; adds an **empty-shelf / out-of-stock** class, which is a real merchandising signal. Proven YOLOv8 graph. **License gate: AGPL-3.0 copyleft** — network-copyleft is the worst fit for a proprietary GTM app (would force source disclosure). | 7 |
| 3 | prince4332/yolov26-product-detection (`best.pt`, YOLO26) | 236 / 0 | **apache-2.0** | small `retail-product-dataset`, 1 class `product` | Ultralytics YOLO → ONNX | **Cleanest license (Apache-2.0)**, but **YOLO26 is a brand-new architecture** → real Melange conversion risk (new/exotic head ops, no PyroGuard-style track record). Unproven quality (0 likes, no published metrics), small training set. | 6 |
| 4 | hatuankiet/YOLOv12S_SKU110K (`yolov12s_best.pt`) | 0 / 0 | mit | SKU-110K, 1 class | Ultralytics YOLO → ONNX | Clean MIT license + SKU-110K task fit, but **YOLOv12 area-attention head is exotic** vs YOLO11 → higher conversion risk; zero traction/metrics; unverified checkpoint. | 5 |
| 5 | sbfisher/yolov8l-sku110k (`best_imgsz640.pt`) | 0 / 0 | none stated | SKU-110K, 1 class | Ultralytics YOLO → ONNX | SKU-110K task fit, proven YOLOv8 graph, but **YOLOv8l (~43M params)** is heavy for a live on-device demo; no license declared (legal ambiguity); zero traction/metrics. | 5 |

(Other repos seen and rejected: `is36e/detr-resnet-50-sku110k` (759 dl) and `benjamintli/dfine-xl_sku110k`,
`benjamintli/rt-detr-v2_sku110k` — DETR/RT-DETR/D-FINE families, **different export family than the
assigned Ultralytics YOLO recipe**, transformer heads are heavier + less Melange-clean; rejected on
family + export-recipe fit. `prince4332/yolov26-product-detection-v2` — same as #3, superseding tag.)

## Winner: chistopat/sku110k-yolo11-object-detector — `weights/sku110k-yolo11-s640.pt` (YOLO11s)

Why this one over the runners-up (Melange-fit + task-fit trade-off):
- **Best Melange-fit in the field.** It is a standard, unmodified **YOLO11s** graph — the exact
  architecture family PyroGuard already ran cleanly through Melange at opset 12. It exports to a
  fixed-shape `[1,3,640,640]` ONNX with only standard ops (verified here: opset 12, `onnx.checker`
  OK, no dynamic axes). The repo even publishes its own fixed-shape ONNX, independent proof the
  graph converts cleanly. The Apache-2.0 alternative (prince4332) is **YOLO26** — a brand-new head
  with real conversion risk; the MIT alternative is **YOLOv12** (exotic area-attention). Proven-arch
  cleanliness beat cleaner licenses here.
- **Best task-fit for the pitch.** SKU-110K is the canonical **dense retail-shelf SKU** dataset —
  one box per product facing on a packed shelf, which is exactly the Trax/Infilect shelf-execution
  problem. Strong metrics (mAP50 0.927 / mAP50-95 0.577). A single-class "object" model that lights
  up a shelf with hundreds of boxes on-device is also the strongest trade-show visual for the
  "boxes drawn on-device, no upload" story.
- **Chosen the S over the N variant** for accuracy headroom (mAP50 0.927 vs 0.906); both are
  lightweight (YOLO11s ≈ 9.4M params, 36 MB ONNX). If the device proves too slow, the drop-in
  lighter same-repo option is `weights/sku110k-yolo11-n640.pt` at the same 640 shape / 1 class.
- **The trade-off accepted — license.** The license is `other`, tied to the upstream **SKU-110K
  dataset terms (research / D&D use)**; the model card explicitly says to keep downstream use aligned
  with SKU-110K terms. For ZETIC's use — an **on-device inference benchmark demo**, not a shipped
  retail-analytics product classifying customer data — this is demo/research use and acceptable, but
  it is a **real GTM flag** if the app is ever productized. Clean-license fallbacks if that day comes:
  `prince4332/yolov26-product-detection` (Apache-2.0, YOLO26 — accept conversion risk) or
  `hatuankiet/YOLOv12S_SKU110K` (MIT, YOLOv12). AGPL-3.0 (foduu) is deliberately avoided as the worst
  license for a proprietary app despite its useful empty-shelf class.

## Export
- Recipe: `export.py` (Ultralytics YOLO → ONNX; same family recipe as PyroGuard).
  `YOLO(sku110k-yolo11-s640.pt).export(format='onnx', imgsz=640, opset=12, simplify=True, dynamic=False, half=False)`
- Input:  `float32[1,3,640,640]`, **NCHW**, values **0.0–1.0** (divide pixels by 255), **RGB** channel order.
- Output: `float32[1,5,8400]`, **channel-major**. Per anchor: `[cx, cy, w, h, object_score]` =
  4 box coords (in **640×640 letterbox pixel space**, verified: box channels max ≈ 638) + 1 class score.
  8400 anchors = 80²+40²+20² across the /8, /16, /32 strides. **No NMS baked in.** The class score is
  **already sigmoid-activated in-graph** (verified range 0.0–1.0 on the raw output) — do NOT re-apply
  sigmoid. YOLO11 is anchor-free with no separate objectness channel (5 = 4 box + 1 class).
- Opset 12; **static shapes confirmed** (no dynamic axes) via `onnx.load` graph inspection and
  `onnxruntime` session I/O; `onnx.checker.check_model` passes.
- Classes (1, verified from checkpoint `model.names`): `object` (a generic retail product facing / SKU).
