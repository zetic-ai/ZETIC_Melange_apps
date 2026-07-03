# Model selection — SafetyPPEYOLO (YOLO object detection, use-case: worker/site-safety PPE detection)

Target: live camera demo detecting person + helmet/hardhat + safety vest (plus
no-helmet / no-vest violation classes) for industrial site-safety / CCTV prospects.

## Shortlist (top 5)

| Rank | HF repo | Downloads | License | Export path | Melange-fit notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1 | ayushgupta7777/safetyvision-yolov8 (v2/best.pt) | 9.5k | AGPL-3.0 (weights) | Ultralytics YOLOv8s -> ONNX, clean | 13 PPE classes incl. Hardhat / NO-Hardhat / Safety Vest / NO-Safety Vest / Person; honest held-out metrics on card (test mAP@0.5 0.754 @640); 44.7 MB ONNX, same scale as PyroGuard | 9/10 |
| 2 | Hexmon/vyra-yolo-ppe-detection | 43.9k | CC-BY-4.0 (weights; YOLOv8 lineage is AGPL) | Ultralytics YOLOv8m -> ONNX, clean | 14 classes, best measured precision — but YOLOv8m: 103.7 MB ONNX / 25.8M params, fails the mobile-size gate; ~2.3x slower | 7/10 |
| 3 | Hansung-Cho/yolov8-ppe-detection | 1.7k | MIT | Ultralytics YOLOv8n -> ONNX, clean | 10 classes incl. all demo classes; tiny (12.2 MB) — but measured precision is demo-unacceptable (person P=0.17, no-vest P=0.17) | 6/10 |
| 4 | leeyunjai/yolo11-ppe (ppe-11s.pt) | 100 | none stated (flag) | Ultralytics YOLO11s -> ONNX, clean | Exactly the 5 target classes (helmet/no-helmet/vest/no-vest/person); same author as PyroGuard's model — but measured accuracy mediocre across the board (micro R 0.37) | 5/10 |
| 5 | keremberke/yolov8n-hard-hat-detection | 1.3k | none stated (ultralyticsplus era) | Ultralytics YOLOv8n -> ONNX, clean | Solid pedigree but only 2 classes (Hardhat / NO-Hardhat) — no vest, no person; task fit too narrow | 5/10 |

Also head-to-head-tested beyond the shortlist top-3: melihuzunoglu/ppe-detection
(YOLO11n, 4 classes — weak everywhere, micro R 0.34) and
badmuriss/ppe-detection-yolov8s (2 classes helmet/vest, CCTV-tuned — helmet great
R 0.87 but vest degenerate on our set, R 0.065). Both rejected on measurement.
DETR-family candidates (hwm21/detr-resnet-50-hardhat-*, MarianaMCruz/detr-finetuned-ppe)
were excluded pre-shortlist: different export recipe (violates the family-recipe batch
rule), heavier runtime, and weaker documented metrics. qualcomm/PPE-Detection hosts no
loadable weights (release_assets.json only).

## VALIDATION-GATED SELECTION — head-to-head results (the deciding evidence)

Shared ground-truth test set: 40 images / 303 labeled boxes from
**keremberke/construction-safety-object-detection** (HF dataset, Roboflow
"Construction Site Safety" derived; CC BY 4.0 per Roboflow universe listing) —
all qualifying images from its test+valid splits plus vest-heavy train-split
images to balance classes. GT box counts: helmet 152, no-helmet 24, vest 46,
no-vest 28, person 53.

Pipeline = exact app pipeline: letterbox 640x640 (gray pad 0.5), /255, NCHW,
ONNX via onnxruntime CPU, channel-major decode, conf 0.25, per-class NMS IoU 0.45,
greedy matching at IoU 0.5.

Per-class P/R at conf 0.25:

| Model | helmet | no-helmet | vest | no-vest | person | micro P/R | ms/img (M3 CPU) |
|---|---|---|---|---|---|---|---|
| **safetyvision_v2s (winner)** | **.79/.86** | .78/.29 | **1.00/.26** | 1.00/.07 | —/.00 | **.80/.50** | 81 |
| hexmon_m | .84/.85 | .75/.13 | .92/.48 | .50/.07 | —/.00 | .84/.52 | 186 |
| hansung_n | .62/.60 | .48/.63 | .34/.30 | .17/.64 | .17/.55 | .33/.55 | 27 |
| leeyunjai_11s | .76/.43 | .53/.33 | .30/.15 | .21/.32 | .21/.43 | .40/.37 | 61 |
| melih_11n | .64/.43 | .14/.21 | .21/.22 | —/.00 (no no-vest class) | .21/.42 | .35/.34 | 30 |
| badmuriss_8s | .75/.87 | n/a | 1.00/.07 | n/a | n/a | .76/.45 | 77 |

Threshold sweep (0.15 / 0.25 / 0.40) confirmed the ranking is stable; lowering the
winner's conf to 0.15 lifts vest recall to 0.35 at P 0.94 — recommended per-class
thresholds for the app: helmet 0.25, vest / violation classes 0.15.

## Winner: ayushgupta7777/safetyvision-yolov8 (v2/best.pt, YOLOv8s)

Why this one over the runners-up:
- Measured, not reasoned: ties hexmon_m (the only accuracy peer) on demo-critical
  helmet/vest precision while being **2.3x smaller (44.7 vs 103.7 MB ONNX)** and
  2.3x faster — hexmon_m is a YOLOv8m and fails the mobile-size criterion.
- Every recall-oriented rival (hansung_n, leeyunjai_11s, melih_11n) has
  demo-unacceptable precision (0.14-0.62 on key classes): wrong boxes on a live
  booth camera are worse than occasional misses.
- Clean static export on the family recipe, opset 12, checker-verified,
  onnxruntime-verified; honest and detailed model card with held-out test metrics.

## Measured weaknesses — read before building (loud and honest)

1. **Person class is DEGENERATE.** 0 person predictions across all 40 images
   (53 GT persons), even at conf 0.05, confirmed both via our ONNX pipeline and
   native Ultralytics predict. The app/UI must NOT rely on person boxes. The demo
   is helmet/vest compliance detection, not person detection. (hexmon_m has the
   same dead class; only the low-precision models fire on person at all.)
2. **Vest recall is modest far-field** (0.26 @ conf .25, 0.35 @ .15) though
   precision is ~perfect. Expected to improve at close/booth range (larger
   objects) — see demo_validation/overlay_img_027.jpg where roofer vests are hit
   at .45-.74 — but unproven until the on-device run.
3. **Violation-class recall is low far-field** (NO-Hardhat 0.29, NO-Safety Vest
   0.07). Close-range behavior is much better: overlay_img_023.jpg nails
   no-helmet 0.87 / helmet 0.88 at booth distance — this is the demo's money shot.
4. **Possible eval contamination**: all viable candidates are Roboflow-derived and
   may share source imagery with the GT dataset; the comparison is fair across
   candidates but absolute numbers may be optimistic.

## License posture (flag to human)

- Winner weights: **AGPL-3.0** on the HF card. Also, ALL candidates are Ultralytics
  YOLOv8/11 fine-tunes — Ultralytics code is AGPL-3.0, so the entire family shares
  this posture regardless of the weight license claimed (hexmon's CC-BY-4.0 weight
  claim sits on the same AGPL architecture/training stack). For an internal
  trade-show demo (not distributed software) risk is low, but shipping this app
  publicly would trigger AGPL obligations or require an Ultralytics commercial
  license. Surfacing, not deciding — human call at GATE 0.

## Export

- Recipe: family recipe (PyroGuard) — `YOLO(path).export(format='onnx', imgsz=640,
  opset=12, simplify=True, dynamic=False, half=False)`; see export.py.
- Input:  float32[1,3,640,640], NCHW, RGB, 0.0-1.0 (/255), letterbox 640 pad 0.5.
- Output: float32[1,17,8400], channel-major; per anchor [cx,cy,w,h, 13 class
  scores]; coords in 640-letterbox space; scores sigmoid-applied; **NMS NOT baked
  in** (verified: raw score range 0-0.12 on random input, 8400 anchors intact).
- Opset 12; static shapes confirmed via onnx.checker + graph inspection (no
  dynamic axes); runs under onnxruntime with sample_input.npy.
- Classes (id order): 0 Fall-Detected, 1 Gloves, 2 Goggles, 3 Hardhat, 4 Mask,
  5 NO-Gloves, 6 NO-Goggles, 7 NO-Hardhat, 8 NO-Mask, 9 NO-Safety Vest,
  10 No_Harness, 11 Person, 12 Safety Vest.
