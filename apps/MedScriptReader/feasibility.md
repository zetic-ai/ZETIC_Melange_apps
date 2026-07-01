# MedScriptReader — Feasibility Pre-Check (Stage 0, GATE-0 blocked pending human go/no-go)

**Assignment:** OCR (medical vision-OCR). Upload a photo of a prescription / medication
label / handwritten clinical note; extract drug name, dosage, frequency, instructions
into structured fields. Candidate model: **naazimsnh02/medocr-vision**. Target sector:
healthcare / India lead. On-device justification (PHI/HIPAA) is real and strong.

**This is a feasibility pre-check only. No ONNX export, no Flutter code, no upload.**

---

## 1. Architecture finding (decisive)

`naazimsnh02/medocr-vision` is **NOT a single-ONNX fixed-output OCR model.** It is a
**1-billion-parameter autoregressive Vision-Language Model (VLM)** — the exact
encoder-decoder + tokenizer + generation-loop pattern the assignment flagged as
highest-risk, and at a scale far beyond the Whisper-decoder analogy.

Concrete config evidence (from the repo's `config.json`, verified locally via
`AutoConfig`/`huggingface_hub`):

| Field | Value |
|---|---|
| `model_type` | `paddleocr_vl` |
| `architectures` | `["PaddleOCRVLForConditionalGeneration"]` (…`ForConditionalGeneration` = seq2seq generation) |
| Base model | `unsloth/PaddleOCR-VL` (1.0B params), LoRA fine-tune merged back into a single 1.92 GB `model.safetensors` |
| Vision encoder | `PaddleOCRVisionModel`, **NaViT-style *dynamic-resolution*** encoder, 27 layers, hidden 1152, patch 14, 384×384 base |
| Text decoder | ERNIE-4.5-0.3B-style causal LM, 18 layers, hidden 1024, **GQA** (16 heads / 2 KV heads), vocab **103,424**, RoPE θ=500k with **3D NeXt-style RoPE scaling** |
| Generation | `model.generate()` autoregressive loop, **KV cache**, `max_new_tokens`, BOS/EOS/pad + image/video/vision-start special tokens |
| Tokenizer | SentencePiece (`tokenizer.model` 1.61 MB) + `tokenizer.json` 11.2 MB, ~103k vocab, chat template (`chat_template.jinja`) |
| Custom code | `trust_remote_code` required: `modeling_paddleocr_vl.py` (111 KB) + `configuration_paddleocr_vl.py`; **requires `einops`** at import (non-standard tensor ops) |

**Export verdict:** This is a two-(really three-)part chain — dynamic-resolution vision
encoder → 103k-vocab autoregressive text decoder with a KV-cache loop → SentencePiece
detokenizer. It is **NOT single-ONNX-clean.** A faithful export would be a multi-model
ONNX bundle plus a full autoregressive decode loop and a 103k-vocab SentencePiece
tokenizer re-implemented in Dart on the worker thread.

## 2. License

**MIT License** (stated on the model card). Commercial/demo use is permitted — this is
the *one* clean dimension. License is NOT the blocker here. (Note: the base
`unsloth/PaddleOCR-VL` / underlying PaddleOCR-VL is Apache-2.0-family; nothing
restrictive surfaced. Confirm the base-model license text before any GTM ship, but no
red flag found.)

## 3. Print vs handwriting

Model card says it handles **both** printed and handwritten medical documents (training
mix: 1,000 prescriptions [handwritten], 426 lab reports [printed], 36 OMR, 1,000
invoices/receipts — 2,462 samples total). The card itself carries the caveat:
**"extremely poor handwriting may not be accurately recognized."** For a healthcare
prospect where a misread dosage is worse than no demo, handwriting is the fragile path.

## 4. Real-image validation (ran the model in PyTorch/transformers, NOT via export)

Toolchain: own uv venv (Python 3.12), `transformers==4.56.2` (the remote code targets
4.56; transformers 5.x rejected the custom config), `torch` CPU, `torchvision`, `einops`,
`sentencepiece`. Loaded via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
(auto_map registers `PaddleOCRVLForConditionalGeneration` under AutoModel/AutoModelForCausalLM),
fp32 on CPU, greedy decode, `max_new_tokens=200`.

**Test images (2 real handwritten + 1 clean printed):**
- **PRINTED (synthetic, known ground truth):** a clean medication label —
  "AMOXICILLIN 500 mg / Capsules Qty:21 / Take ONE capsule by mouth / THREE times a day
  for 7 days / Take with food. Finish all. / Rx# 4821990 Dr. R. Sharma".
- **HANDWRITTEN rx_00006** (real, from HF dataset `chinmays18/medical-prescription-dataset`,
  faint cursive). GT: doctor `Dr. C. Rossi`, clinic `City Health Clinic`, `45 Oak Ave.`,
  patient `Aisha Khan`, age `73`, date `2024-12-16`, med `Prednisone 25 mg – After meals`.
- **HANDWRITTEN rx_00025** (same source). GT: `Dr. F. Gomez`, `Greenwood Medical Center`,
  `234 Maple Blvd.`, patient `Wei Li`, age `41`, date `2024-12-16`, med
  `Metformin 250 mg – After meals`.

**Runtime observation (material to the verdict):** loading + generating on a multi-core
desktop CPU consumed **~35 min of CPU-time in ~13 min wall-clock**, ~4 GB RSS, for just
these few images. A 1B fp32 VLM autoregressive decode is nowhere near interactive
latency; on a phone this is not a real-time or even a few-seconds-per-scan experience.

**Actual model outputs (greedy, fp32 CPU) vs ground truth — the model reads WELL:**

| Image | CPU latency | Model output | Accuracy vs GT |
|---|---|---|---|
| PRINTED label | 176 s | "AMOXICILLIN 500 mg Capsules - Qty: 21 Take ONE capsule by mouth THREE times a day for 7 days Take with food. Finish all. Rx# 4821990 Dr. R. Sharma" | **Verbatim exact — 100%.** |
| HANDWRITTEN rx_00006 | 495 s | "45 Oak Ave. Prescribed by: Dr. C. Rossi Date: 2024-12-16 Patient: Aisha Khan Age: 73 **Predinisone** 25 mg - After meals Signature: Dr. C. Rossi" | Doctor/address/patient/age/date/freq/signature all correct. **Dropped clinic name** ("City Health Clinic"). **Drug misspelled: "Predinisone" ≠ "Prednisone".** |
| HANDWRITTEN rx_00025 | 592 s | "234 Maple Blvd. Prescribed by: Dr. F. Gomez Date: 2024-12-16 Patient: Wei Li Age: 41 **Metformin 250 mg** - After meals Signature: Dr. F. Gomez" | All fields correct incl. drug+dosage. **Dropped clinic name** ("Greenwood Medical Center"). |

**Read-quality takeaway:** the OCR is genuinely good — it read faint cursive handwriting
and recovered drug / dosage / frequency on both prescriptions (Metformin 250 mg exact;
Amoxicillin printed exact). Two failure signatures appeared: (a) it systematically
**dropped the top clinic-name header line** on both handwritten scans, and (b) it
**misspelled a drug name** ("Predinisone" for "Prednisone") — which is precisely the
patient-safety failure that makes a *handwritten*-prescription demo risky for a
healthcare prospect. So the *model* is not the weak link; the **deployability form
factor** is.

## 5. Melange-fit assessment (EXPLORATION.md §4 rubric)

| Rubric criterion | Verdict for medocr-vision |
|---|---|
| Exportable to ONNX | Only as a multi-model chain; no clean single-graph path. **Fail (as a demo artifact).** |
| Static-shape friendly | NaViT **dynamic resolution** + autoregressive KV-cache loop = inherently dynamic axes. **Fail.** |
| Standard ops (opset-12) | Custom remote code, `einops`, 3D RoPE, GQA, VLM cross-modal fusion. opset-12 clean convert is not realistic. **Fail.** |
| Mobile-sized | **1.92 GB / 1.0B params.** Rubric wants single-digit-to-low-tens of MB. **Fail by ~100×.** |
| License | MIT. **Pass.** |
| Quality/popularity | Niche fine-tune, low downloads, single author. Weak signal. |
| Task fit | Genuinely medical OCR incl. handwriting — good task fit, wrong *form factor* for Melange. |
| Output format known | Free-form generated text (structured-ish `<s_ocr> key: value` schema). Not a fixed tensor. |

**Export-shape reality if forced:** encoder ONNX (dynamic image grid) + decoder ONNX with
external KV-cache I/O looped from Dart + a Dart 103k-vocab SentencePiece tokenizer +
greedy/beam decode + `<s_ocr>` field parsing. That is a large worker-side build and
fights every one of Melange's static-shape / small-model / standard-op assumptions.
This is categorically heavier than PyroGuard (single static YOLO graph, NMS in Dart).

**Lower-risk alternative (recommended path to the same use-case):** use the same clean
**two-stage OCR recipe** the other apps can share — a text **detector** (e.g. DB/PP-OCR
det) + a compact **printed-text recognizer** (PP-OCRv4 / SVTR / CRNN), both of which
export to static-shape ONNX at tens of MB, run per-crop (no autoregressive loop), and hit
the **clean printed medication-label / lab-report** use-case reliably. That keeps the
healthcare/PHI on-device story intact for the demoable surface (printed labels) without
betting a healthcare prospect on a 1B VLM reading cursive.

## 6. Recommendation

**Verdict: NO-GO for this batch (deferred to a separate future run).**
`naazimsnh02/medocr-vision` is a **1.0B-param autoregressive Vision-Language Model**
(`PaddleOCRVLForConditionalGeneration`, 1.92 GB). It fails the Melange-fit rubric on every
form-factor axis: **exportability** (a multi-model encoder → KV-cache decoder chain, no
clean single-graph path), **static shapes** (dynamic-resolution NaViT encoder + an
autoregressive decode loop), **standard ops** (custom remote code, `einops`, 3D RoPE, GQA),
and **mobile size** (~100× the budget) — and it won't run at demo latency on-device.
License (MIT) and read-quality are fine; the **form factor is the blocker.**

**Why deferred rather than forced:** the model actually reads well — it recovered printed
text verbatim (Amoxicillin label 100%) and read faint cursive handwriting, getting
drug/dosage/frequency right on both prescriptions. But it **dropped the clinic-name header
on both handwritten scans** and **misspelled a drug name ("Predinisone" for "Prednisone")**
— a patient-safety failure mode that is unacceptable for a healthcare demo. And even setting
accuracy aside, the on-device form factor (1B fp32 VLM, autoregressive KV-cache decode,
seconds-to-minutes per scan) is simply **unshippable through Melange.** Both reasons point
the same way.

**Recommended path for a FUTURE run (not this batch):** if the medical-OCR use-case is
revisited, use the same **two-stage PP-OCR recipe** the other three apps use — a text
**detector** (DB / PP-OCR det) + a compact **printed-text recognizer** (PP-OCRv4 / SVTR /
CRNN), both static-shape opset-12 ONNX at tens of MB, scoped to **clean printed medication
labels / lab reports**. That keeps the PHI/on-device story without betting a healthcare
prospect on a 1B VLM reading cursive. Note this would be a **fresh Stage-0 exploration**,
not a continuation of this one.

**Status:** **DROPPED from the current OCR batch by human decision on GATE-0 feasibility
review.** No ONNX export, no Melange upload, no HANDOFF ticket.
