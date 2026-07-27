# SPEC: ContentModeration

> Stage-0 pre-drafted spec stub. Everything the locally-verified ONNX contract fixes is
> concrete. Melange model name/version are the only late-binding fields.

## One-line pitch
On-device NSFW / content-safety gate: classify a user image (or video frame) fully on
the phone before upload — no cloud call — and drive a keep / blur / block decision. Built
for Trust & Safety buyers (dating, social, marketplace, moderation vendors).

## Model
- Source (HF repo / origin): Marqo/nsfw-image-detection-384 (Apache-2.0)
- Architecture: timm `vit_tiny_patch16_384` (ViT-Tiny, patch16, 384px), 5.6M params, ~22.5 MB ONNX
- Melange model name:  ajayshah/ContentModeration  **[LATE-BINDING — placeholder until GATE-0 paste-back]**
- Melange version:     1  **[LATE-BINDING — placeholder until GATE-0 paste-back]**
- Input tensor: float32[1,3,384,384], NCHW, RGB, value range [-1,1] after normalization (see pre-processing)
- Output tensor: float32[1,2] RAW LOGITS. Semantic layout: index 0 = NSFW logit, index 1 = SFW logit (config label_names = ["NSFW","SFW"]).
- Post-processing baked into ONNX? No. No softmax, no activation, no NMS baked in — raw logits out.
- Classes / labels: ["NSFW", "SFW"]  (idx0 = NSFW, idx1 = SFW)
- modelMode to use and why: RUN_AUTO. This is a ViT (attention heads = the exact fusion pattern that hit the iOS-26 MPSGraph GPU crash on PyroGuard). No client modelMode avoids that crash; it is handled server-side by ZETIC filtering the GPU candidate for affected OS. Read the SERVED artifact (target+apType) from the native console; realistic non-crashing fallback is CPU (TFLITE_FP16), not the NPU.

## Input source
- File / gallery pick and/or camera frame (single-image moderation is the core demo; a live-camera "frame safety meter" is an optional mode).
- Pixel format: cheapest usable (BGRA on iOS / YUV420 on Android) — this is a full-frame classifier, not a detector, so orientation is far less critical than PyroGuard, but still resize from the true buffer.
- Orientation handling: a full-image classifier is largely orientation-robust; still resize/crop from the correctly-oriented buffer. No box overlay to mis-rotate.

## Pre-processing pipeline (ordered, exact) — MUST match timm eval transform exactly
1. Decode frame/image to RGB (drop alpha; convert BGRA/YUV -> RGB).
2. Resize so the SHORTEST edge = 384, preserving aspect ratio. Interpolation = **BICUBIC with antialiasing** (timm uses bicubic, antialias=True). Getting this wrong (bilinear / nearest / no-antialias) shifts scores near the decision boundary — see Validation focus.
3. Center-crop 384 x 384.
4. Convert to float32 and rescale * 1/255  -> [0,1].
5. Normalize per channel (v - 0.5) / 0.5  -> [-1,1]  (mean = std = [0.5,0.5,0.5], all 3 channels).
6. Reorder HWC -> CHW, add batch -> [1,3,384,384], RGB channel order. Flatten to Float32List, wrap as `Tensor.float32List(data, shape:[1,3,384,384])`.

## Post-processing pipeline (ordered, exact)
1. Read output `logits` = Float32List length 2, order [NSFW, SFW].
2. Apply softmax over the 2 logits: `P = softmax([l_nsfw, l_sfw])`.
3. `pNsfw = P[0]` (== 1 - P[1]).
4. Decision bands (defaults; expose as tunable thresholds in UI):
   - `pNsfw < 0.30`            -> KEEP (auto-approve)
   - `0.30 <= pNsfw < 0.70`    -> REVIEW / BLUR (soft action / send to human review)
   - `pNsfw >= 0.70`           -> BLOCK (do not upload)
5. Emit a Result{ pNsfw, pSfw, decision, band-color }.
(No coordinate spaces, no letterbox inverse, no NMS — this is whole-image classification.)

## UI
- Left to the worker. Functional must-haves: show the picked/captured image; the decision
  (KEEP / REVIEW-BLUR / BLOCK) with band color; P(NSFW) and P(SFW) as a live meter/scores;
  an inference-latency readout; a tunable threshold control is a nice-to-have. Consider a
  blur-overlay preview for the REVIEW/BLOCK bands (that IS the demo story).

## Platform targets
- iOS minimum: 16.6+ ; Android minSdk 24 (match repo convention).
- Known OS traps for this model/artifact: ViT attention -> FP32-GPU CoreML artifact can
  crash in MPSGraph on iOS/macOS 26.3+ (SIGABRT, uncatchable in Dart). Not client-fixable;
  ZETIC filters GPU server-side for affected OS. Always read the served target+apType from
  the native console; expect CPU (TFLITE_FP16, hundreds of ms) as the realistic fallback,
  NPU/Neural-Engine only if ZETIC serves a CoreML NE artifact.

## Validation focus (Tier A traps most likely for THIS model)
- **Score semantics / activation.** Output is RAW LOGITS — a softmax MUST be applied; skipping it or applying sigmoid silently corrupts confidence. Test: hand-built logits -> known softmax -> known P(NSFW).
- **Label order.** index 0 = NSFW, index 1 = SFW (NOT the usual "0=safe"). A swapped index inverts every decision. Test with a hand-built [big, small] vs [small, big] logit pair.
- **Normalization exactness.** (x-0.5)/0.5 -> [-1,1], NOT a plain /255 [0,1]. Test the preprocessing on a known constant image (e.g. all-127 pixels -> ~0.0 tensor).
- **Resize interpolation.** Must be bicubic + antialias to match the eval transform; a bilinear/nearest resize measurably shifts borderline scores (ONNX-vs-reference Δ was ~0.007 mean but up to 0.22 on one high-frequency borderline image, traced to resize/antialias). Test a known image round-trips to the reference tensor within tolerance.
- **Channel order.** RGB, not BGR — decode buffers accordingly. Test with a single-color image.
- **Decision-band thresholds.** Test the 0.30 and 0.70 boundaries (just-below / just-above land in the right band).
- No letterbox/NMS/orientation-overlay traps (whole-image classifier, no boxes).
