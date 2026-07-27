# LiveGate demo images — measured pipeline results

All scores below were produced by running the **shipped ONNX artifacts**
(`../face_sface.onnx`, `../pad_minifasnet_v2.onnx`) in onnxruntime with the EXACT
preprocessing the app will use. Images are selected BY MEASURED OUTPUT (not eyeballed).

Provenance: faces are from public research face sets of public figures / celebrities
(LFW = Labeled Faces in the Wild; CelebA-Spoof) — not private individuals' photos.
They are included only as a pipeline-verification fixture for the worker/orchestrator.

## Slot 1 — Face 1:1 match (SFace, cosine of L2-normalized 128-d embeddings)

| Image(s) | Scenario | Cosine | Verdict @ thr 0.363 |
|---|---|---|---|
| `face_ref_personA.png` vs `face_probe_personA.png` | **same person** | **0.423** | ✅ MATCH |
| `face_ref_personA.png` vs `face_probe_personB.png` | **different people** | **0.213** | ❌ NO-MATCH |

Clear separation (same 0.423 >> diff 0.213). Absolute cosine is modest because these
demo crops use a crude CENTER-CROP, not landmark alignment — on-device 5-point
alignment lifts same-person cosine substantially (see model_selection.md aggregate).

## Slot 2 — Liveness / PAD (MiniFASNet-V2, LIVE = softmax[class 1])

| Image | Crop fed to model | live-score | Verdict @ thr 0.45 |
|---|---|---|---|
| `pad_live_realface.png` | `pad_live_crop.png` | **1.000** | ✅ LIVE |
| `pad_spoof_attack.png`  | `pad_spoof_crop.png` | **0.000** | 🚫 SPOOF |

The `*_crop.png` files show the exact 2.7× face-margin crop that is resized to 80×80
and fed to the model — this crop geometry is load-bearing (see SPEC_STUB.md); a tight
crop breaks the model. These two are the cleanest-separating pair in the sample; the
honest aggregate over 60 live / 60 spoof is ~81% (see model_selection.md), so the live
demo should use a real printed-photo / phone-screen spoof, which separates strongly.
