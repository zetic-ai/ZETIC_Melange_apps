# Model selection — RetinaDRScreen (medical-imaging classification, use-case: on-device diabetic-retinopathy screening of color fundus images)

Sector: point-of-care DR screening (targets AEYE Health / AEYE-DS, RETINA-AI Health).
The on-device wedge: fundus images are screened on the handheld device and never leave
it, so it works offline in low-connectivity / low-end screening camps. NEVER re-sell
FDA clearance — on-device changes data-residency / offline posture only, not clinical
claims.

Single-model Melange demo — this app wraps exactly ONE ONNX, one forward pass:
fundus image in -> 2 referable-vs-not logits out.

## ⚠️ This REPLACES the previous pick

The earlier selection (`Aniket-1234/mobilenetv3-diabetic-retinopathy`, a 5-grade
MobileNetV3) was **degenerate on validation** and has been **discarded** (its ONNX
`mobilenetv3-dr-grading.onnx` is deleted from this folder). To choose a trustworthy
replacement, a **6-way validation bakeoff** was run on a shared, grade-labeled fundus
set (42 images spanning grade 0-4, drawn from IDRiD + APTOS; artifacts in `_eval/*/`).
The winner is a compact BINARY referable screener that is validated, not degenerate.

## The bakeoff — all 6 candidates

Evaluated on the same 42-image set (GT grades 0-4). "Referable" = DR grade >= 2
(Moderate+), the standard screening threshold. 5-grade / regression models are mapped
to referable by thresholding their predicted grade; the winner has a NATIVE binary head.
`g0->not-ref` = how many of the 6 healthy (grade-0) eyes were correctly called
not-referable — the key anti-degeneracy check.

| Rank | HF repo | Arch | ONNX size | License | Ref. sens | Ref. spec | Binary acc | g0->not-ref | 5-grade exact |
|------|---------|------|-----------|---------|-----------|-----------|------------|-------------|---------------|
| **1 — SELECTED** | **EscvNcl/MobileNet-V2-Retinopathy** | **MobileNetV2-1.4, NATIVE binary NRDR/RDR** | **17.4 MB** | **`other` (flag)** | **0.833** | **0.889** | **0.857** | **6/6** | n/a (binary) |
| 2 | Kontawat/vit-diabetic-retinopathy-classification | ViT-base, 5-grade | 343 MB | apache-2.0 | 1.000 | 0.833 | 0.929 | 5/6 | 0.667 |
| 3 | Augusto777/swinv2-tiny-...-Diabetic-Retinopathy | SwinV2-tiny, 5-grade | 112 MB | apache-2.0 | 1.000 | 0.667 | 0.857 | 5/6 | 0.524 |
| 4 | vyshnav112233/diabetic-retinopathy-efficientnet-b5 | EfficientNet-B5, REGRESSION head | 113 MB | none declared (flag) | 0.917 | 0.611 | 0.786 | 5/6 | 0.452 |
| 5 | its-karthick1/dr-grading-effv2l | EfficientNetV2-L, 5-grade | 470 MB | mit | 1.000 | 0.056 | 0.595 | 1/6 | 0.119 |
| 6 | rafalosa/diabetic-retinopathy-224-procnorm-vit | ViT-base (procnorm), 5-grade | 343 MB | apache-2.0 | 0.875 | 0.722 | 0.810 | 5/6 | 0.381 |

Reading the table:
- **#5 EfficientNetV2-L is degenerate** — it flags almost everything referable
  (spec 0.056; only 1/6 healthy eyes called not-referable). High sensitivity is
  meaningless when specificity collapses; this is the failure mode the bakeoff exists
  to catch. #4 (EffB5 regression) is a milder version of the same over-flagging.
- **#2 ViT-base (Kontawat) has the best raw binary accuracy (0.929)** and full
  sensitivity, but it is a **343 MB ViT** — 20x the winner's size, transformer
  attention that fights Melange's compile step and lives in the iOS-26 MPSGraph
  GPU-crash family, and a 5-grade model bent into a binary decision by thresholding.
- **#3 SwinV2 and #6 rafalosa-ViT** are mid-pack: decent sensitivity, weaker
  specificity, and again large transformer graphs (112-343 MB).

## Winner (SELECTED at GATE 0): EscvNcl/MobileNet-V2-Retinopathy

Best balance of Melange-fit and task-fit **for THIS app** (a tiny, offline,
low-end-device referable screener):

- **Tiny + ideal Melange target.** ~17 MB ONNX — by far the smallest in the bakeoff
  (the next-smallest is 112 MB). A standard mobile CNN: the exported opset-12 graph is
  ordinary ops (Conv/Add/Clip/Relu6/GlobalAveragePool/Gemm) — no attention, no dynamic
  axes. This is exactly the graph Melange compiles cleanly, and it dodges the ViT /
  MPSGraph GPU-crash class that the 343 MB transformer candidates carry. Best fit for
  the low-connectivity / low-end screening-camp pitch.
- **Perfect on healthy eyes (6/6).** The ONLY candidate that called all 6 grade-0 eyes
  not-referable. On real referable screening, over-flagging healthy patients is the
  costly error (needless referrals swamp the clinic); the winner is the least prone to
  it (spec 0.889, best in the field).
- **Native binary output matches the product.** id2label = {0 Nrdr, 1 Rdr} — a genuine
  NRDR-vs-RDR head, not a 5-grade softmax hacked into a threshold. That is precisely the
  referable / not-referable readout AEYE-DS / RETINA-AI-style autonomous screening
  produces — one forward pass, one decision, no severity bookkeeping.
- **Validated, not degenerate.** Referable sensitivity 0.833 / specificity 0.889 /
  binary accuracy 0.857 across the 42-image set. Correct trade-off for a screener: it
  slightly under-calls a few borderline referable cases rather than over-flagging the
  healthy majority.

Trade-off accepted vs #2 ViT: the ViT edges it on raw binary accuracy (0.929 vs 0.857)
and sensitivity (1.0 vs 0.833). We deliberately take the compact native-binary CNN here
because device-fit (17 MB, no attention, no MPSGraph risk) and healthy-eye specificity
are the deciding axes for THIS on-device screener. **The heavyweight ViT 5-grade
severity option is carried by the sibling app `RetinaDRGrade`** — so the product line
covers both: a tiny screener here, a richer severity-grader there.

Caveat (must survive to the SPEC / demo): this is a BINARY screener — it gives
referable / not-referable, NOT a 0-4 severity grade. Metrics are on 42 research-dataset
images (IDRiD/APTOS), not a clinical population. It is a capability/latency demo of
on-device screening, NOT a validated diagnostic device — never represent it as
diagnostic or FDA-cleared. Validate on-device before any screening claim (VALIDATION.md
Tier C).

## ⚠️ LICENSE — pre-ship legal check (flagged, not resolved)

`EscvNcl/MobileNet-V2-Retinopathy` declares **`license: other` with NO stated terms**.
The base model `google/mobilenet_v2` is Apache-2.0, but the fine-tuned DR weights'
redistribution / commercial terms are **UNDECLARED**. This is a hard GTM gate: before
shipping, get the license clarified (author / training-data terms). If it cannot be
cleared, the drop-in alternates from this same bakeoff are the Apache-2.0 transformers
(#2 Kontawat ViT, #3 Augusto SwinV2, #6 rafalosa ViT) or the MIT #5 (EffV2L, but it is
degenerate) — each far larger and a worse Melange fit, so clearing the winner's license
is the preferred path.

## Export
- Recipe: `export.py` — medical-imaging-classification family. Load the transformers
  `MobileNetV2ForImageClassification`, wrap to return raw logits, `torch.onnx.export`
  (opset 12, `dynamo=False`, `do_constant_folding=True`, STATIC [1,3,224,224],
  `half=False`, NO dynamic axes). Clean exported ONNX + eval also in
  `_eval/mnv2-escvncl/`.
- Artifact: `mobilenetv2-dr-referable.onnx` (~17.4 MB).
- Input:  float32 `pixel_values` **[1,3,224,224]**, NCHW, RGB. Preprocessing (NOT plain
  /255): resize shortest-edge->256 (bilinear) -> center-crop 224 -> *1/255 ->
  normalize (v-0.5)/0.5 with mean=std=[0.5,0.5,0.5] -> NCHW. The Dart preprocessor owns
  this exactly.
- Output: float32 `logits` **[1,2]**, RAW LOGITS. **No softmax baked in** — apply
  softmax in Dart; P(referable) = softmax[index 1]; referable if P(idx1) >= threshold
  (default 0.5). id2label = {0 Nrdr (not-referable), 1 Rdr (referable)}.
- Opset 12. **Static shapes confirmed** (onnx.checker passes; dynamic axes = False; op
  set is all standard CNN ops). torch-vs-onnxruntime parity verified at export.
