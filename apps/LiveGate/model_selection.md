# Model selection — LiveGate (face-biometrics family, use-case: on-device KYC live-selfie gate)

Two-model app: (1) a PAD / anti-spoof classifier + (2) a face-recognition embedding model.
Both slots were **VALIDATION-GATED** — the actual exported ONNX was run in onnxruntime with
the exact app preprocessing against labeled public data, and winners were crowned on measured
numbers, not shortlist reasoning.

Target sector: Vision / KYC — identity-verification vendors (e.g. Onfido/Entrust, Jumio,
Incode / Socure-style IDV players).

---

## SLOT 1 — PAD / anti-spoof classifier

### Shortlist (top 5)
| Rank | HF repo | DLs | License | Export path | Melange-fit notes | Score |
|------|---------|-----|---------|-------------|-------------------|-------|
| 1 | **garciafido/minifasnet-v2-anti-spoofing-onnx** | 0 | **Apache-2.0** | pre-exported ONNX (opset 11→12, pin batch) | 1.7 MB; std Conv/BN/PRelu; classic mobile PAD family; **only slot-1 candidate that is BOTH ONNX + permissive + validatable** | **8.5** |
| 2 | litert-community/Silent-Face-Anti-Spoofing-LiteRT | 33 | Apache-2.0 | TFLite only (no ONNX) | same MiniFASNet weights but `.tflite` — Melange wants ONNX; would need TFLite→ONNX | 6.0 |
| 3 | nguyenkhoa/mobilevitv2_Liveness_detection_v1.0 | 55 | other (unclear) | HF transformers → ONNX | 17.6 MB MobileViTv2; attention ops; license "other"/unclear | 5.5 |
| 4 | nguyenkhoa/vit_Liveness_detection_v1.0 | 125 | apache-2.0 | HF ViT → ONNX | **343 MB** ViT-base — far too big for on-device; disqualified on size | 4.0 |
| 5 | rikurunico/minifasnet-onnx / tienminhtran/face-antispoof-onnx | 0 | **none stated** | ONNX | same family but NO license → GTM risk; garciafido supersedes (Apache + card + provenance) | 3.5 |

### Winner: garciafido/minifasnet-v2-anti-spoofing-onnx  (MiniFASNet-V2, 2.7_80x80)
Why over the runners-up (Melange-fit + task-fit + license):
- **Apache-2.0 and provenanced** to minivision-ai Silent-Face-Anti-Spoofing (bit-equivalent
  weights, SHA-256 in the card). The TFLite mirror is the same model but wrong format; the
  license-less MiniFASNet mirrors are a GTM risk; the ViT/MobileViT liveness models are
  either too big (343 MB) or license-murky.
- Tiny (1.7 MB), pure conv/BN/PRelu/Gemm → clean, fully static ONNX for Melange.
- It is the classic, battle-tested mobile silent-PAD family the assignment called out.

### Measured validation (⚠️ read the caveats — PAD confidence is MODERATE, stated plainly)
Data: `nguyenkhoa/antispoofing-1` (CelebA-Spoof derivative) — full frames + face bbox +
live/spoof labels. 60 live + 60 spoof, exact 2.7× crop geometry applied.

| Recipe (crop 2.7×, 80×80) | live-img mean softmax[1] | spoof-img mean softmax[1] | best acc |
|---|---|---|---|
| **BGR, range [0,255]** (WINNER) | **0.818** | **0.255** | **80.8% @ thr 0.45** |
| RGB, range [0,255] | 0.789 | 0.455 | 70.8% |
| BGR, range [0,1] (÷255) | 0.006 | 0.006 | **50.0% — DEGENERATE (saturated to class 2)** |

**PAD validation confidence: MODERATE.** It is a real, non-degenerate empirical result
(clear live/spoof separation with the correct crop), but note honestly:
- **81%** is aggregate over CelebA-Spoof's *diverse* spoof mix (print, replay, 3D mask, etc.).
  Upstream MiniVision ensembles TWO models (V1SE@4.0 + V2@2.7); we ship ONE, so this
  single-model number is expected to be lower than the ensemble. For the specific demo
  (real face vs a printed photo / phone-screen replay) separation is much cleaner — the
  curated demo pair scores live=1.000 vs spoof=0.000.
- **⚠️ TWO model-card errors were caught by validation — trust the ONNX, not the card:**
  1. **Input range:** the card says `pixel/255` ([0,1]); the ONNX SATURATES to a constant
     class-2 output at [0,1] and only discriminates on **raw BGR [0,255]**. The app MUST
     feed [0,255].
  2. **Live class:** the card says classes are `[live, print, replay]` (live=class 0);
     empirically **LIVE = class 1** (MiniVision `test.py` convention). class 0/2 are spoof.
- **Crop geometry is load-bearing:** feeding the dataset's tight face crop (instead of the
  2.7× margin crop) collapses the model to constant output. Get the crop right or scores
  are meaningless (see SPEC_STUB.md for the exact recipe).

### Export
- Recipe: `export.py` → download ONNX, pin batch=1, onnxslim-fold (Shape/Gather vanish),
  opset 11→12. FP32 (Melange owns precision).
- Input:  float32 **[1,3,80,80]**, NCHW, **BGR**, **range [0,255] (NOT /255)**.
- Output: float32 **[1,3]** raw logits → softmax; **LIVE = softmax[1]**; spoof = 0/2.
- Post-processing baked in? No (softmax is pure-Dart). Static shapes confirmed, opset 12.

---

## SLOT 2 — Face-recognition embedding

### Shortlist (top 5)
| Rank | HF repo | DLs | License | Export path | Melange-fit notes | Score |
|------|---------|-----|---------|-------------|-------------------|-------|
| 1 | **opencv/face_recognition_sface** | 0 | **Apache-2.0** | pre-exported ONNX (strip weight-inputs, opset 11→12) | 38.5 MB; std Conv/BN/PRelu/Gemm; [1,3,112,112]→[1,128]; norm baked in; **commercial-clean** | **8.7** |
| 2 | InsightFace **w600k_mbf** (buffalo_s, MobileFaceNet) | high | **❌ non-commercial research ONLY** | ONNX in buffalo_s.zip | 13 MB, [1,3,112,112]→[1,512], ~99.7% LFW — best quality BUT **license sinks a GTM demo** | 6.5 |
| 3 | anjith2006/edgeface (+ Idiap/EdgeFace-*) | 72 | **❌ CC-BY-NC-SA-4.0** | safetensors → ONNX | 5–15 MB, edge-purpose-built, strong — but **non-commercial** | 6.0 |
| 4 | gaunernst/vit_tiny_patch8_112.arcface_ms1mv3 | 16573 | none stated | timm → ONNX | 22 MB ViT-tiny; attention ops; license unstated + MS-Celeb-1M-derived data provenance risk | 5.5 |
| 5 | py-feat/mobilefacenet | 22 | none stated (py-feat=MIT toolbox) | .pth → custom ONNX | 12 MB MobileFaceNet; needs custom export harness; weight license unclear | 5.0 |

### Winner: opencv/face_recognition_sface  (SFace, MobileFaceNet-style ArcFace)
Why over the runners-up (the license axis decided it — as the assignment demands):
- **Apache-2.0** = GTM-clean, with a bundled LICENSE and a published paper (arXiv:2205.12010).
  The two best-quality mobile alternatives (InsightFace **w600k_mbf** and **EdgeFace**) are
  BOTH **non-commercial** — flagged LOUDLY here — which disqualifies them for a GTM demo
  even though w600k_mbf is smaller (13 MB) and slightly more accurate.
- Already ONNX, already static [1,3,112,112]→[1,128], all standard ops, OpenCV-maintained.
- Normalization `(x-127.5)/128` is **baked into the graph**, so the app just feeds raw
  0..255 BGR — one less place to get wrong on-device.
- 38.5 MB is at the upper end of "mobile-sized" but well within the low-tens-of-MB budget.

### Measured validation
Data: `logasja/lfw` (Labeled Faces in the Wild), 12 identities with ≥3 images each,
central-crop approximation of alignment, cosine of L2-normalized embeddings.

| Metric | Value |
|---|---|
| # same-identity pairs / # different-identity pairs | 120 / 380 |
| cosine, **same** identity | mean **0.505**, std 0.126, min 0.215 |
| cosine, **different** identity | mean **0.166**, std 0.103, max 0.449 |
| **best 1:1 verification accuracy** | **94.8% @ cosine thr 0.390** |
| @ thr 0.363 | acc 94.0%, same-accept 84.2%, impostor-accept (FPR) **2.9%** |
| @ thr 0.400 | acc 94.8%, same-accept 82.5%, impostor-accept (FPR) **1.3%** |

**Face validation confidence: HIGH.** Clear, non-degenerate same-vs-different separation.
94.8% is a floor, achieved with a CRUDE center-crop; SFace reports ~99.4% LFW with proper
ArcFace 5-point alignment — so on-device alignment (worker does landmark alignment app-side)
will raise this. Recommended demo threshold **0.363–0.40** (low impostor-accept for a KYC gate).

### Export
- Recipe: `export.py` → download ONNX, strip the 174 weight tensors from `graph.input`
  (leave only `data`), onnxslim-fold, opset 11→12. FP32.
- Input:  float32 **[1,3,112,112]**, NCHW, **BGR**, **range [0,255]** (in-graph (x-127.5)/128).
- Output: float32 **[1,128]** embedding, **NOT L2-normalized in-graph** → normalize in Dart,
  then cosine (dot of unit vectors). Post-processing baked in? No. Static shapes, opset 12.
