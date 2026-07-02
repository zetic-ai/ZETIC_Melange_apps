# Demo images — RetinaDRGrade

Three fundus images chosen to DEMONSTRATE DISCRIMINATION across the referable boundary:
one confident correct **No DR (grade 0, not referable)** and two confident correct
**referable (grade >= 2)** cases spanning **Severe (3)** and **Proliferative (4)**. Source
images copied (not moved) from `../RetinaDRScreen/demo_images/candidates/`. Each has a
`*_viz.png` (fundus + 5-grade softmax bar + GT vs predicted grade), regenerable with
`validate_demo.py`.

Grades: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative. Referable = grade >= 2.

## Per-image measured output (exported ONNX, argmax == grade)
| Image | GT grade | Predicted grade | Top prob | Full softmax [0,1,2,3,4] | Referable (pred) | Correct? |
|-------|----------|-----------------|----------|--------------------------|------------------|----------|
| IDRiD_g0_6389f96a.png | 0 No DR | **0 No DR** | 0.982 | [0.982, 0.008, 0.005, 0.002, 0.003] | no  | yes |
| IDRiD_g3_dd7d2789.png | 3 Severe | **3 Severe** | 0.810 | [0.007, 0.006, 0.088, 0.810, 0.089] | yes | yes |
| IDRiD_g4_278b9ee5.png | 4 Proliferative | **4 Proliferative** | 0.809 | [0.019, 0.011, 0.035, 0.125, 0.809] | yes | yes |

Demo subset: 3/3 exact-grade correct; referable sens 1.00, spec 1.00 (tp=2 fn=0 tn=1 fp=0).
These probabilities reproduce the full-eval `predictions.json` exactly, which also confirms
the pure-numpy preprocessing in `validate_demo.py` matches HF's ViTImageProcessor.

## Aggregate model accuracy (full 42-image held-out eval, IDRiD + APTOS)
- Exact-grade accuracy: **0.667** (28/42).
- Referable(>=2) sensitivity: **1.00** (never misses a referable eye).
- Referable(>=2) specificity: **0.833**.
- Per-grade recall: g0 5/6, g1 5/12, g2 12/12, g3 3/6, g4 3/6 — mid-grades (Mild/Severe)
  are the hardest, as expected for DR grading.
(Source: `../RetinaDRScreen/_eval/vit-kontawat/summary.json`.)

## Honest caveats
- **Cross-dataset, research data only.** Eval is on IDRiD + APTOS public research fundus
  images, not the model's training distribution and not a clinical population — real-world
  numbers will differ.
- **Small eval (N=42).** The 3-image demo subset is illustrative, not a benchmark; per-grade
  cells are tiny.
- **NOT a diagnostic device.** This is a GTM/tech demo. It is not FDA/CE cleared, not a
  medical device, and must not be used for diagnosis or treatment decisions.
- **On-device = data-residency only.** Running locally keeps the image on the phone (privacy);
  it does not make the prediction clinically valid.
- **Size:** the ViT-base ONNX is ~343 MB fp32 — a real first-launch download / storage cost.

## Re-run
```
pip install onnxruntime pillow numpy matplotlib
python validate_demo.py     # prints per-image preds + rewrites *_viz.png
```
