# Demo images — ContentModeration

Curated by MEASURED ONNX output (not eyeballed), to show the winner model
(Marqo/nsfw-image-detection-384, exported `nsfw-vit-tiny-384.onnx`) at its best across
the full keep / blur / block decision spectrum. Scores are P(NSFW) = softmax(logits)[0]
from the actual exported ONNX with the exact app preprocessing.

Decision bands: KEEP `P(NSFW) < 0.30` · REVIEW/BLUR `0.30 ≤ P(NSFW) < 0.70` · BLOCK `P(NSFW) ≥ 0.70`.

| Demo | Decision | P(NSFW) | P(SFW) | Source (Wikimedia Commons) |
|------|----------|---------|--------|----------------------------|
| `demo_keep_food` | **KEEP** | 0.056 | 0.944 | "Good Food Display - NCI Visuals Online" (US NCI, public domain) |
| `demo_review_classical_art` | **REVIEW / BLUR** | 0.489 | 0.511 | Botticelli, "La nascita di Venere" (public domain) |
| `demo_block_suggestive` | **BLOCK** | 0.754 | 0.246 | "Bademoden-Modenschau" (swimwear fashion show) |

Each demo ships a clean `<name>.jpg` (the input) and a `<name>_result.png` (annotated card
with decision + scores). Machine-readable scores in `results.json`.

## Honesty notes (per EXPLORATION.md §7)
- The borderline images (classical art, swimwear) are **NON-EXPLICIT proxies**, used only
  to demonstrate the score gradient and the review/block bands. **No explicit imagery was
  collected.** True explicit-content recall is per Marqo's published eval, not measured here.
- The model is a **binary NSFW/SFW** gate. It does not classify violence/gore/weapons as
  separate categories (a legal weapon photo scores P(NSFW)≈0.05 = KEEP, which is correct).
  See `../model_selection.md` for the full honest task-fit discussion.
- Aggregate validation (the real signal, not a highlight reel): 100% SFW specificity on 28
  clearly-safe images, worst-case false-positive P(NSFW)=0.081. See `../model_selection.md`.
- Images are sourced from Wikimedia Commons for validation/demo purposes; verify individual
  file licenses before any public redistribution.
