# Model selection — VehiclePlateYOLO (YOLO object detection, use-case: vehicle / license-plate detection)

Sector: smart-city / parking-management demo (automotive-adjacent, municipal).
Single-model Melange demo — each app wraps exactly ONE ONNX, one forward pass.

## Shortlist (top 5)

Score column reflects the original Melange+task technical ranking. The **SELECTED**
winner is set by the GATE-0 license requirement (MIT mandatory for this GTM app), which
overrides the raw technical top score — see the Winner section.

| Rank | HF repo | Downloads | License | Export path | Melange-fit notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1 — **SELECTED** | **Koushim/yolov8-license-plate-detection** (`best.pt` → `koushim-yolov8-license-plate.onnx`) | 25,310 | **MIT** | Ultralytics `YOLO.export` (identical to PyroGuard recipe) | YOLOv8n, 1 class `license_plate`. Single-stage. Same recipe + same verified I/O as the proven PyroGuard pipeline; static [1,3,640,640]→[1,5,8400]; clean opset-12 ONNX (~12 MB, smallest). **MIT = clean GTM license (the deciding factor).** Thinner model card. | 8.4 |
| 2 — alternative (AGPL, demoted) | morsetechlab/yolov11-license-plate-detection (`license-plate-finetune-v1s.pt`) | 21,656 | AGPL-3.0 | Ultralytics `YOLO.export` (identical to PyroGuard recipe) | YOLO11s, 1 class `License_Plate`. Highest technical score (newer arch, best-documented card, 300 epochs, likely higher recall), but **AGPL-3.0 fails the GTM license gate** so it is demoted, not selected. | **9.0** |
| 3 | yasirfaizahmed/license-plate-object-detection | 874 | Apache-2.0 | Ultralytics `YOLO.export` (yolov8n) | YOLOv8n plate detector, 1 class, 75 epochs @640. Apache-2.0 (commercial-clean) but low popularity/eval signal. Secondary commercial option. | 7.6 |
| 4 | keremberke/yolov5m-license-plate | 23,482 | None declared | YOLOv5 (Ultralytics v5) export — different head/recipe | Popular & well-known, but YOLOv5 = older head, separate export path (not the YOLO11 recipe), no declared license. More conversion risk, GTM license ambiguous. | 6.3 |
| 5 | nickmuchi/yolos-small-finetuned-license-plate-detection | 2,847 | None declared (base hustvl/yolos-small is Apache-2.0) | `transformers` / DETR-style ONNX export | YOLOS = ViT/DETR transformer detector. Exotic attention ops + typically dynamic shapes → fights Melange compile; this is the family the iOS-26 MPSGraph class of bugs lives in. Reject on Melange-fit. | 5.0 |

## Resolving the two-stage trap

Classic ANPR is a **two-stage chain**: detect vehicle → crop → detect plate → OCR.
That is **disqualifying** for a single-model Melange demo, which wraps ONE ONNX run per
app. I explicitly reject any "vehicle detector + separate plate detector" design.

Resolution: pick a **single-stage** detector that emits plate boxes directly in one
YOLO forward pass. The winner detects `license_plate` directly — no vehicle crop, no
second model. A combined *vehicle+plate single-class-set* YOLO would be marginally
richer, but no well-maintained, popular, single-model HF checkpoint for that exists;
the popular "combined" approaches on HF are all two-model chains (disqualified). So a
single-stage plate detector is the strongest true single-model fit.
**OCR of the plate text is out of scope** — that is a separate OCR-family run.

## Winner (SELECTED at GATE 0): Koushim/yolov8-license-plate-detection (`best.pt`)

The license gate is decisive here: this app's GATE-0 requirement is a **commercially
clean MIT license**, so the AGPL morsetechlab model — though it carried the higher raw
technical score — is disqualified for GTM and demoted to a documented alternative. Among
the permissively-licensed candidates, Koushim is the strongest:
- **MIT license** — clean for ZETIC's GTM / trade-show distribution (the deciding factor).
- **Same Melange-fit as the proven path:** Ultralytics YOLOv8, the exact PyroGuard
  export recipe, with VERIFIED I/O matching the family (static [1,3,640,640] →
  [1,5,8400], opset 12, clean onnxslim). Smallest artifact (~12 MB ONNX, ~3M-param nano).
- **Strong popularity:** 25.3k downloads (highest in the field); genuinely single-stage,
  single-class (`license_plate`), fully known I/O.

Trade-off accepted: it is a YOLOv8 **nano** with a thinner model card than morsetechlab's
documented YOLO11s, so expect somewhat lower recall on hard/angled/small plates. That is
the price of the MIT requirement; the morsetechlab AGPL model remains the drop-in
higher-recall fallback (swap `HF_REPO`/`HF_FILE` in `export.py`) **if** the AGPL terms
are ever cleared for this build. Provenance caveat: Koushim declares MIT on weights
trained with AGPL Ultralytics tooling — confirm provenance before shipping as a product.

## Export
- Recipe: `export.py` (Ultralytics YOLO recipe — `format='onnx', imgsz=640, opset=12,
  simplify=True, dynamic=False, half=False`).
- Artifact: `koushim-yolov8-license-plate.onnx` (~12 MB).
- Input:  float32 `images` **[1,3,640,640]**, NCHW, values 0.0–1.0 (divide by 255), RGB.
  (Verified from the exported ONNX.)
- Output: float32 `output0` **[1,5,8400]**, channel-major; per anchor
  `[cx, cy, w, h, plate_conf]`; coords in 640×640 letterboxed space; 8400 anchors over
  80/40/20 grids. **NMS NOT baked in** (no `NonMaxSuppression` node — implement in pure
  Dart). Confidence is already activated: a **`Sigmoid` node is baked into the graph**,
  so apply NO extra sigmoid in Dart. (Both verified by inspecting the exported ONNX.)
- Opset 12. **Static shapes confirmed** (`onnx.checker` passes; no dynamic axes;
  verified by re-reading the exported graph). Single class: `["license_plate"]`.
