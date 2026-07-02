# DentalXrayDetect — Demo Images (validated against ground truth)

These demo panoramic radiographs were **measured**, not eyeballed. Each was run through the
exported ONNX (`dentalxray-yolo11n.onnx`) with the exact SPEC_STUB pipeline, and every predicted
box was matched against DENTEX ground-truth annotations at IoU ≥ 0.50 (same class). Reproduce with
`validate_demo.py` (see repro command at the bottom).

- **Model:** `dentalxray-yolo11n.onnx` — YOLO11n, `liodon-ai/dental-panoramic-detector`, 3 classes
  `{0 caries, 1 periapical_lesion, 2 impacted_tooth}`, trained on DENTEX + OralXrays-9 panoramics.
- **Inference settings (model-card recommended, per SPEC_STUB):** conf **0.45**, per-class NMS IoU **0.45**,
  imgsz 640 letterboxed, scores taken as-is (already sigmoid'd in-graph — no re-sigmoid).
- **Ground-truth source:** DENTEX **challenge validation split** (`validation_triple.json`), the
  quadrant-enumeration-**disease** set — 50 panoramic X-rays, 182 abnormal-tooth annotations across
  46 images. DENTEX diagnosis labels mapped to model classes:
  `Caries + Deep Caries → caries`, `Periapical Lesion → periapical_lesion`, `Impacted → impacted_tooth`.
- **Overlay legend:** yellow = ground truth (GT:label), green = predicted box that matched a GT
  (true positive, shows `conf` + `IoU`), red = predicted box with no GT match (false positive).

## Source & license (all three images)

- **Source URL:** https://huggingface.co/datasets/ibrahimhamamci/DENTEX
  (file path inside the repo: `DENTEX/validation_data.zip → validation_data/quadrant_enumeration_disease/xrays/<name>.png`;
  GT in `DENTEX/validation_triple.json`).
- **Split:** DENTEX challenge **validation** split (quadrant-enumeration-disease). Held out from the
  main training set; used by the model card as its eval set (`mAP50 0.622 / mAP50-95 0.406`). Treat
  as validation-grade, not a fully independent test set — see caveats.
- **License:** **CC-BY-NC-SA-4.0** (non-commercial, share-alike). Fine for an internal
  capability-proof demo; **cannot ship in a commercial product**. Attribution: Hamamci et al.,
  DENTEX (MICCAI 2023), grand-challenge.org/dentex.

---

## 1. `val_28.png` — HERO: all three pathology classes correct in one radiograph
- **Original resolution:** 2872 × 1504
- **GT present:** 4 Impacted, 4 Caries, 1 Deep Caries, 2 Periapical Lesion (11 annotations)
- **Model detections @ conf 0.45 (8 total, 7 true positives):**

  | class | conf | IoU→GT | GT match | TP? |
  |---|---|---|---|---|
  | caries | 0.810 | 0.95 | Deep Caries | ✅ |
  | impacted_tooth | 0.782 | 0.86 | Impacted | ✅ |
  | impacted_tooth | 0.782 | 0.81 | Impacted | ✅ |
  | impacted_tooth | 0.752 | 0.76 | Impacted | ✅ |
  | impacted_tooth | 0.748 | 0.73 | Impacted | ✅ |
  | caries | 0.600 | 0.91 | Caries | ✅ |
  | periapical_lesion | 0.487 | 0.74 | Periapical Lesion | ✅ |
  | caries | 0.588 | 0.00 | — | ❌ (1 FP) |

- **Why this photo:** the single most complete proof — the model correctly boxes **all three of its
  classes** on one image (caries IoU 0.95 @ 0.81 conf, four impacted molars, and a periapical lesion),
  each landing tightly on the real pathology. One caries false positive is present and honestly shown in red.

## 2. `val_38.png` — CLEAN impacted_tooth demo (zero false positives)
- **Original resolution:** 2942 × 1316
- **GT present:** 4 Impacted, 1 Periapical Lesion (5 annotations)
- **Model detections @ conf 0.45 (4 total, 4 true positives, 0 FP):**

  | class | conf | IoU→GT | GT match | TP? |
  |---|---|---|---|---|
  | impacted_tooth | 0.808 | 0.85 | Impacted | ✅ |
  | impacted_tooth | 0.769 | 0.83 | Impacted | ✅ |
  | impacted_tooth | 0.764 | 0.80 | Impacted | ✅ |
  | impacted_tooth | 0.747 | 0.70 | Impacted | ✅ |

- **Why this photo:** four impacted wisdom teeth, all boxed tightly, **no false positives** — the
  cleanest, most legible overlay. `impacted_tooth` is the model's strongest class (recall 0.82) and
  this image shows it unambiguously. (The 1 periapical GT here is not detected — shown honestly as an
  un-boxed yellow region.)

## 3. `val_32.png` — CLEAN caries demo (zero false positives)
- **Original resolution:** 2747 × 1316
- **GT present:** 2 Caries, 2 Deep Caries (4 annotations)
- **Model detections @ conf 0.45 (4 total, 4 true positives, 0 FP):**

  | class | conf | IoU→GT | GT match | TP? |
  |---|---|---|---|---|
  | caries | 0.777 | 0.86 | Deep Caries | ✅ |
  | caries | 0.704 | 0.90 | Caries | ✅ |
  | caries | 0.693 | 0.92 | Deep Caries | ✅ |
  | caries | 0.645 | 0.90 | Caries | ✅ |

- **Why this photo:** every carious lesion in the image is caught with high IoU (0.86–0.92) and
  **no false positives** — a rare clean caries result that reads clearly. Balances the set so caries
  (the model's hardest class) also has a convincing standalone example.

---

## Aggregate performance across the FULL tested set (honest numbers)

All 46 annotated DENTEX validation images, conf 0.45 / NMS IoU 0.45 / match IoU 0.50:

| class | GT | TP | recall | preds | FP | precision |
|---|---|---|---|---|---|---|
| caries | 133 | 46 | **0.35** | 81 | 35 | 0.57 |
| periapical_lesion | 9 | 3 | **0.33** | 3 | 0 | 1.00 |
| impacted_tooth | 40 | 33 | **0.82** | 44 | 11 | 0.75 |
| **TOTAL** | 182 | 82 | **0.45** | 128 | 46 | 0.64 |

Reference point at conf 0.25 (why we do **not** lower the threshold): caries recall rises to 0.68 but
caries precision collapses to **0.33** (275 predictions, 184 false positives) — the model over-fires
caries on adjacent healthy teeth, exactly as the model card warns. Conf 0.45 is the right operating point.

**Reading of these numbers:**
- **impacted_tooth** is the genuinely strong class: recall 0.82, precision 0.75, high-confidence tight boxes.
- **caries** is a screening hint, not a count: it finds ~1 in 3 lesions but is fairly precise (0.57) when it fires.
- **periapical_lesion** is data-starved (only 9 GT instances) — perfect precision but low recall; treat any single result as anecdotal.
- The three demo images above are the cases where the model **genuinely nails it**; they are not representative of average recall, which is ~0.45 overall.

## Caveats (must accompany any use of these images)

- **Capability proof, NOT a diagnostic device.** These overlays demonstrate on-device detection
  capability only. Nothing here is clinically validated and **none of it implies or confers FDA
  clearance**. On-device deployment changes data-residency only, never regulatory status.
- **Panoramic-only.** Weights are trained on panoramic radiographs and validated here on panoramic
  radiographs — that is the model's real capability. It is **unproven on bitewing / periapical
  intra-oral films** and should not be claimed to work on them without re-validation/fine-tuning.
- **Thin overall accuracy.** DENTEX mAP50 0.622 / mAP50-95 0.406. Do not overclaim: caries recall is
  ~0.35, periapical is data-starved. The demo set is deliberately the model's best, legible cases.
- **Validation-split, not fully independent test.** The DENTEX validation split doubles as the model
  card's eval set; there is no guarantee it was fully excluded from every training run in the model's
  lineage. Numbers are validation-grade, treated conservatively.
- **License:** CC-BY-NC-SA-4.0 — non-commercial internal demo use only.

## Reproduce

```bash
# from apps/DentalXrayDetect/ (deps: onnxruntime numpy pillow opencv-python matplotlib huggingface_hub datasets)
python validate_demo.py                         # full-set scoring table + candidate ranking (conf 0.45)
python validate_demo.py --conf 0.25             # threshold-sensitivity check
python validate_demo.py --overlay val_28.png    # (re)render an overlay into demo_images/
```
Ground truth: `work_val_triple.json` (copy of DENTEX `validation_triple.json`).
Images: `work_val/` (extracted from `validation_data.zip`).
