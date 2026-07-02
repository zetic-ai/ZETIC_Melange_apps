# Model selection — DentalXrayDetect (medical-imaging detection, use-case: dental x-ray caries / bone-loss / periapical detection)

Target sector: chairside dental AI. Prime target **Overjet** (FDA-cleared caries outlining +
bone-level quantification, currently cloud SaaS) — the wedge is **on-device** so radiographs/PHI
never leave the practice. Also VideaHealth. On-device changes DATA RESIDENCY only, never FDA status.

Core task-fit point: the use-case wants **pathology** detection (caries / periapical lesion /
bone-loss) on dental radiographs, not mere tooth enumeration. Purpose-built, permissively-licensed
caries/bone-loss detectors are scarce on the Hub — most public dental models are either tooth-number
(FDI) detectors, undocumented single-class hobby exports, or LLM/text noise. Pathology-class coverage
with real eval numbers is therefore treated as a first-class task-fit factor below.

## Shortlist (top 5)

| Rank | HF repo | DL / Likes | License | Arch / size | Export path | Melange-fit + task-fit notes | Score /10 |
|------|---------|-----------|---------|-------------|-------------|------------------------------|-----------|
| 1 | **liodon-ai/dental-panoramic-detector** | 125 / 0 | **cc-by-nc-4.0 (NON-COMMERCIAL ⚠️)** | YOLO11n, 5.2 MB pt / 10 MB onnx | Ultralytics YOLO → ONNX, standard ops | Only shortlisted model with real **pathology classes** (caries, periapical_lesion, impacted_tooth) AND real eval (mAP50 0.622, mAP50-95 0.406 on DENTEX val), documented I/O, recommended thresholds, per-class notes, ships a reference ONNX. Standard tiny YOLO11n → cleanest Melange fit. **Sole negative: NC license (GTM gate).** | **9** |
| 2 | Sentoz/dental-opg-cavity-detection-model | 0 / 0 | **mit** ✅ | YOLOv8n, 22.5 MB pt | Ultralytics YOLO → ONNX | Clean commercial license + right task (cavity/caries, 640). BUT README eval is a placeholder (`mAP 0.0 # update after training`), single class, undocumented custom dataset → quality unverified. **The license-clean fallback if GTM needs commercial rights.** | 6 |
| 3 | sanleigo/caries_detection_v1 | 0 / 0 | **wtfpl** ✅ (max-permissive) | ~YOLO, 22.5 MB pt | Ultralytics YOLO → ONNX | Maximally permissive license and caries task, but **empty README** — no classes, no imgsz, no eval, no dataset. Unauditable output format = downstream guess. Permissive but opaque. | 5 |
| 4 | Gaurav2k/dentex-yolo11x-v2-best | 0 / 0 | none stated | YOLO11**x**, 114 MB pt | Ultralytics YOLO → ONNX | Trained on DENTEX (caries/disease) with a data.yaml, so task-fit is real — but YOLO11x is ~40× the params of 11n: heavy for a live on-device demo, and no license is a GTM blocker. | 5 |
| 5 | abychkov/dental-fdi-detection | 0 / 0 | other (restrictive) | Transformer (DETR-style), 131 MB onnx | Ships ONNX | Professional, 4000+ radiographs, CPU-tuned — but it detects **32 FDI tooth numbers, not pathology** (wrong task), it is a **transformer** detector (exotic-op / attention conversion risk vs a clean CNN YOLO), 131 MB (large), and license is "other". Off-task + Melange-risky. | 4 |

(Also seen and rejected: `Mohitha/yolov8-bone-loss` & `chemahc94/pano-boneloss-weights` — right
task (bone-loss) but no license and no documentation/eval; `liodon`-style pathology coverage is
stronger. `LazerX69/Dental-anomalies-yolov8` — "abood"/other license, no eval. `prakash1702`,
`MTAR1`, `LadySakura`, `joshuarebo` — large undocumented YOLO exports, no license/eval. The many
`*-Gensyn-Swarm-*toothy*`, GGUF, Qwen/Gemma/GPT2 hits are LLM/text-to-image noise, not detectors.)

## Winner: liodon-ai/dental-panoramic-detector

Why this one over the runners-up (Melange-fit + task-fit trade-off):
- **Only genuine pathology detector in the field.** It is the sole shortlisted model that outputs
  the exact classes the use-case (and the Overjet wedge) care about — **caries** and
  **periapical_lesion** — plus impacted_tooth, with **published eval numbers** (mAP50 0.622) and
  documented, reproducible I/O. Every license-cleaner alternative is either single-class,
  undocumented, or has a placeholder/zero eval. Task-fit + auditable output dominated here.
- **Best Melange fit of the whole field.** Standard unmodified **YOLO11n** (2.58 M params, 6.3
  GFLOPs, 10 MB ONNX) — the same clean, static-shape, standard-op export as PyroGuard/AerialDetect.
  It is also the smallest credible option → ideal for NPU on-device. (Runner-up Gaurav is YOLO11x
  ~40× larger; abychkov is a transformer with attention-op conversion risk.)
- **The trade-off accepted — LICENSE (loud flag):** `cc-by-nc-4.0` is **NON-COMMERCIAL**. It is
  fine for an internal capability-proof demo (the stated purpose — "use public dental
  datasets/models for the capability proof"), but it **cannot ship in a commercial product**. The
  drop-in, license-clean fallback is **Sentoz/dental-opg-cavity-detection-model (MIT)** — same
  Ultralytics YOLO export recipe, same imgsz 640 — at the cost of unverified quality (placeholder
  eval, single class). If GTM needs commercial rights, swap and re-register; the app pipeline is
  unchanged (both are YOLO detect heads).
- **The other trade-off — domain shift:** the weights are trained on **panoramic** radiographs, but
  the target modality is **bitewing / periapical**. The README itself notes caries recall is
  "limited at panoramic resolution." This is a capability proof, not a production bitewing model; a
  production build would fine-tune on bitewing/periapical data. Treat caries output as a screening
  hint, not a count. (This is a data/quality caveat, not a conversion blocker.)
- **Clinical honesty:** on-device changes data-residency only; it does NOT confer/alter FDA
  clearance. Never claim otherwise in the demo.

## Export
- Recipe: `export.py` (Ultralytics YOLO → ONNX; same family recipe as PyroGuard/AerialDetect, imgsz 640).
  `YOLO(best.pt).export(format='onnx', imgsz=640, opset=12, simplify=True, dynamic=False, half=False)`
  (We re-export from `best.pt` rather than use the repo's shipped `best.onnx`, to guarantee opset 12 +
  static axes.)
- Input:  `float32[1,3,640,640]`, NCHW, RGB channel order, values **0.0–1.0** (divide pixels by 255).
- Output: `float32[1,7,8400]`, channel-major. Per anchor: `[cx, cy, w, h, s0, s1, s2]` =
  4 box coords (**pixel space** in the 640×640 letterbox frame) + 3 class scores. 8400 anchors =
  80²+40²+20² across /8, /16, /32 strides. **NMS is NOT baked in.** Class scores **ARE
  sigmoid-activated in-graph (already 0–1)** — do NOT re-apply sigmoid in Dart. Verified from the
  exported ONNX via onnxruntime (box rows range ~5.8–636 = pixels; class rows within [0,1]).
- Opset 12; **static shapes confirmed** (no dynamic axes) by inspecting the exported ONNX graph.
- Classes (3, verified from checkpoint `model.names`): `caries, periapical_lesion, impacted_tooth`.
- Recommended inference settings (from model card): conf 0.45, iou 0.35, imgsz 640. At conf 0.25 the
  model over-fires caries on adjacent teeth.
