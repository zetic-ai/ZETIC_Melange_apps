# Model selection — AerialDetectYOLO (YOLO object detection, use-case: drone / aerial top-down detection)

Target sector: industrial inspection, agriculture (drone surveillance / inspection demo).

Core task-fit point: a generic COCO YOLO is trained on ground-level, eye-height objects and
performs poorly on small, top-down aerial targets. Every shortlisted model is therefore trained
on an **aerial/drone dataset (VisDrone-2019)**, not COCO. Training dataset is treated as a
first-class task-fit factor below.

## Shortlist (top 5)

| Rank | HF repo | Downloads (mo) / Likes | License | Training dataset | Export path | Melange-fit notes | Score /10 |
|------|---------|------------------------|---------|------------------|-------------|-------------------|-----------|
| 1 | **ENOT-AutoDL/yolov8s_visdrone** (`baseline_enot/weights/best.pt`) | 196 / 11 | **Apache-2.0** | VisDrone-2019, **imgsz 928** | Ultralytics YOLO → ONNX, standard YOLOv8s ops | Clean commercial license; standard YOLOv8s graph; aerial-trained at high res → best small-object recall. Repo also ships `baseline_ultralytics` (imgsz 640) as a lighter same-license option. Avoid the `enot_neural_architecture_selection_x2/x3` weights — NAS-modified architectures = conversion risk. | **9** |
| 2 | mshamrai/yolov8s-visdrone | 636 / 6 | OpenRAIL | VisDrone-2019, imgsz 640 | Ultralytics YOLO → ONNX | Most-downloaded VisDrone YOLOv8; clean YOLOv8s graph, easy 640 export. mAP@0.5 0.408. **License gate: OpenRAIL carries behavioural use-restrictions** — flag for GTM. | 7 |
| 3 | mshamrai/yolov8m-visdrone | 336 / 1 | OpenRAIL | VisDrone-2019, imgsz 640 | Ultralytics YOLO → ONNX | Higher accuracy (mAP@0.5 0.454) but YOLOv8m is ~3× the params of 8s — heavier on NPU for a live demo. Same OpenRAIL flag. | 6 |
| 4 | mshamrai/yolov8l-visdrone | 375 / 5 | OpenRAIL | VisDrone-2019, imgsz 640 | Ultralytics YOLO → ONNX | mAP@0.5 0.451; YOLOv8l is large (~43M params) — overkill / heavy for on-device. Same OpenRAIL flag. | 5 |
| 5 | mshamrai/yolov8n-visdrone | 332 / 1 | OpenRAIL | VisDrone-2019, imgsz 640 | Ultralytics YOLO → ONNX | Smallest/fastest (nano) but lowest accuracy (mAP@0.5 0.341) — weak on the small aerial objects that are the whole point. Same OpenRAIL flag. | 5 |

(Other repos seen: `Mahadih534/YoloV8-VisDrone` — CC license, only person/vehicle, fewer classes;
`qualcomm/YOLOv8-Detection` — generic **COCO** weights, NOT aerial-trained, rejected on task-fit.)

## Winner: ENOT-AutoDL/yolov8s_visdrone — `baseline_enot/weights/best.pt`

Why this one over the runners-up (Melange-fit + task-fit trade-off):
- **License is the deciding gate.** It is the only Apache-2.0 VisDrone YOLO in the field; every
  mshamrai variant is OpenRAIL (behavioural use-restrictions that complicate a commercial GTM
  demo). For ZETIC's go-to-market motion, Apache-2.0 is a clean, unambiguous yes.
- **Aerial small-object fit.** The `baseline_enot` checkpoint was trained at **imgsz 928**, not
  640. Higher input resolution is exactly what top-down drone imagery needs (objects are tiny),
  giving the best reported small-object accuracy (repo mAP@0.5 ≈ 0.494) of the shortlist.
- **Melange-cleanliness.** It is an *unmodified* standard YOLOv8s graph (only the training recipe
  differs), so it exports to ONNX with the same standard-op recipe as PyroGuard — no exotic ops.
  The NAS-optimized siblings in the same repo were deliberately avoided to keep ops conventional.
- **The trade-off accepted:** 928×928 input is ~2.1× the pixels of 640×640, so NPU/CPU latency
  will be higher than a 640 model. That is a runtime (Tier C) cost, not a conversion blocker. If
  the device proves too slow, the drop-in lighter option is the **same repo, same Apache-2.0
  license**: `baseline_ultralytics/weights/best.pt` exported at imgsz 640 (identical 10 classes).
- **Commercial runner-up (different license):** if ENOT is ever unavailable, mshamrai/yolov8s-visdrone
  is the popularity leader but is OpenRAIL — usable for a demo, but surface the use-restrictions.

## Export
- Recipe: `export.py` (Ultralytics YOLO → ONNX; same family recipe as PyroGuard, imgsz changed).
  `YOLO(best.pt).export(format='onnx', imgsz=928, opset=12, simplify=True, dynamic=False, half=False)`
- Input:  `float32[1,3,928,928]`, NCHW, values **0.0–1.0** (divide pixels by 255), RGB channel order.
- Output: `float32[1,14,17661]`, channel-major. Per anchor: `[cx, cy, w, h, p0..p9]` =
  4 box coords (in 928×928 letterbox space) + 10 class scores. 17661 anchors = 116²+58²+29²
  across the /8, /16, /32 strides. **No NMS and no sigmoid baked in** (raw YOLOv8 anchor-free
  head; YOLOv8 has no separate objectness — 14 = 4 + 10). Class scores are raw and already
  in [0,1] from training but post-processing should threshold on max class score.
- Opset 12; **static shapes confirmed** (no dynamic axes) by inspecting the exported ONNX graph.
- Classes (10, verified from the checkpoint `model.names`, VisDrone order):
  `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`.
