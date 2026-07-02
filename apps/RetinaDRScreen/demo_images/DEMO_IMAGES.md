# RetinaDRScreen — demo images & measured model behavior

Model: `mobilenetv2-dr-referable.onnx` — MobileNetV2-1.4 **binary referable screener**
(`EscvNcl/MobileNet-V2-Retinopathy`, license **`other` — see caveats**). This is NOT a
5-grade severity model. It outputs 2 raw logits; `id2label = {0: Nrdr (not-referable),
1: Rdr (referable, DR grade ≥ 2)}`. Decision: **referable if P(index 1) ≥ 0.5**.

Preprocessing used (exactly per `SPEC_STUB.md` / the model's `preprocessor_config.json` —
NOT plain /255): RGB → **resize shortest-edge → 256** (bilinear) → **center-crop 224** →
`× 1/255` → **normalize `(v − 0.5)/0.5`** (mean = std = `[0.5,0.5,0.5]`) → NCHW
`float32[1,3,224,224]`. Output = raw logits `[1,2]` → **softmax** → `P(referable) = softmax[1]`.
Every number below is a **measured ONNX output vs. ground truth**, not eyeballed.
Reproduce with `python validate_demo.py` (from the app root).

> ⚠️ **This is a research/demo capability check, NOT a diagnosis and NOT a validated
> clinical device.** On-device inference changes data-residency / offline posture only,
> never clinical validity. Do not represent it as FDA-cleared or diagnostic.

---

## Selected demo images (3) — chosen to DEMONSTRATE DISCRIMINATION

The set spans the referable decision the model exists to make: one **confident healthy
(grade-0) eye → NOT-REFERABLE** with near-zero P(referable), and two **confident
referable (grade-3 / grade-4) eyes → REFERABLE** with near-one P(referable). All three
are from **IDRiD** (CC-BY-4.0). Each has a rendered `*_viz.png` (fundus + P(referable)
bar with the 0.5 threshold marked + predicted decision + GT grade) alongside it.

| File | GT grade | Referable GT | Measured P(referable) | Logits [Nrdr, Rdr] | Decision | Correct | viz |
|---|---|---|---|---|---|---|---|
| `IDRiD_g0_630e24b6.png` | 0 (No DR) | not-referable | **0.0000** | [10.11, −0.66] | **NOT REFERABLE** | ✅ | `demo_healthy_g0_IDRiD_g0_630e24b6_viz.png` |
| `IDRiD_g3_ca10d891.png` | 3 (Severe) | referable | **0.9958** | [−2.72, 2.75] | **REFERABLE** | ✅ | `demo_severe_g3_IDRiD_g3_ca10d891_viz.png` |
| `IDRiD_g4_ce3e6abe.png` | 4 (Proliferative) | referable | **0.9900** | [−2.30, 2.29] | **REFERABLE** | ✅ | `demo_proliferative_g4_IDRiD_g4_ce3e6abe_viz.png` |

The healthy eye is called not-referable with essentially 0 probability of disease while
the severe/proliferative eyes are flagged referable at ~0.99 — clean, confident
discrimination across the threshold, not a stuck classifier.

## Aggregate binary metrics (full 42-image bakeoff eval)

Measured on 42 labeled fundus images spanning grades 0–4 (IDRiD + APTOS), referable =
DR grade ≥ 2. Source: `_eval/mnv2-escvncl/results.json`.

| Metric | Value |
|---|---|
| Referable **sensitivity** | **0.833** (TP=20, FN=4) |
| Referable **specificity** | **0.889** (TN=16, FP=2) |
| **Binary accuracy** | **0.857** (36/42) |
| Healthy (grade-0) called not-referable | **6/6** |

This model won a 6-way validation bakeoff (see `../model_selection.md`) — it is the
smallest artifact (~17 MB), has the best healthy-eye specificity in the field, and is
the only candidate that got all 6 grade-0 eyes right.

## Honest caveats

- **Binary only, no severity.** It outputs referable / not-referable, NOT a 0–4 grade.
  Do not surface a severity grade in the app. (The 5-grade ViT severity option lives in
  the sibling app RetinaDRGrade.)
- **License `other` (undeclared).** `EscvNcl/MobileNet-V2-Retinopathy` declares
  `license: other` with no stated terms; the base `google/mobilenet_v2` is Apache-2.0
  but the fine-tuned weights' terms are unresolved. Pre-ship legal check required (see
  `../melange_upload.md`).
- **Research datasets.** Metrics are on 42 public research-dataset images (IDRiD/APTOS),
  not a clinical population; small-sample, not a generalization guarantee. It slightly
  under-calls a few borderline referable cases (sensitivity 0.833) — acceptable for a
  screener that must not over-flag the healthy majority, but validate on-device
  (VALIDATION.md Tier C) before any screening claim.
- **On-device ≠ diagnostic.** Running offline on the handset changes data-residency
  only. This is a capability/latency demo, NOT a diagnostic or FDA-cleared device.

## Data sources & licenses
- **IDRiD** (all 3 demo images) — Indian Diabetic Retinopathy Image Dataset, research
  use, **CC-BY-4.0** on the HF mirror used to assemble `candidates/`. Labels match the
  canonical grade order.
- **APTOS 2019** — used only as additional labeled test data in the 42-image aggregate,
  not shipped as a demo image; underlying APTOS 2019 is research-only.
- All images are labeled research datasets — fine for an internal demo; **not for
  redistribution or any clinical claim.**

## Repro / provenance
- `validate_demo.py` (app root) — standalone reproducer: runs the ONNX with the exact
  preprocessing on the 3 demo images, renders the `*_viz.png` panels, writes
  `results.json`.
- `results.json` — per-image GT, logits, P(referable), decision + the aggregate metrics.
- `candidates/` — the 42 labeled fundus images (shared across the bakeoff).
- Full 6-way bakeoff artifacts: `../_eval/*/`. Winner's clean eval + ONNX + export
  script: `../_eval/mnv2-escvncl/`.
