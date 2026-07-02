#!/usr/bin/env python3
"""
validate_demo.py — RetinaDRScreen demo-image validator (re-runnable).

Runs the SERVED-equivalent ONNX (`mobilenetv2-dr-referable.onnx`) on a handful of
labeled fundus images with the EXACT MobileNetV2 preprocessing the model expects,
computes P(referable) via softmax, and renders one `<name>_viz.png` per image
(fundus + P(referable) bar + predicted decision + GT grade). Also writes
demo_images/results.json.

The demo set is chosen to DEMONSTRATE DISCRIMINATION: a confident healthy grade-0
eye -> NOT-REFERABLE (low P), and confident referable grade-3/4 eyes -> REFERABLE
(high P).

Preprocessing (must match export.py / SPEC_STUB.md exactly — NOT plain /255):
  resize shortest-edge -> 256 (bilinear) -> center-crop 224 -> *1/255 ->
  normalize (v-0.5)/0.5 (mean=std=[0.5,0.5,0.5]) -> NCHW [1,3,224,224], RGB.

Labels: id2label = {0: "Nrdr" (not-referable), 1: "Rdr" (referable)}.
Decision: referable if P(index 1) >= THRESHOLD (default 0.5).

Run:  python validate_demo.py     (needs: onnxruntime, numpy, pillow, matplotlib)
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ONNX = HERE / "mobilenetv2-dr-referable.onnx"
CAND = HERE / "demo_images" / "candidates"
OUT = HERE / "demo_images"
THRESHOLD = 0.5

# Demo set (from demo_images/candidates/) chosen to show discrimination:
#   one confident HEALTHY grade-0 -> not-referable, two confident REFERABLE grade-3/4.
DEMO = [
    ("IDRiD_g0_630e24b6.png", "healthy"),         # grade 0 — expect NOT-REFERABLE, low P
    ("IDRiD_g3_ca10d891.png", "severe"),          # grade 3 — expect REFERABLE, high P
    ("IDRiD_g4_ce3e6abe.png", "proliferative"),   # grade 4 — expect REFERABLE, high P
]


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # resize shortest edge -> 256, preserve aspect (bilinear)
    if w <= h:
        nw, nh = 256, round(256 * h / w)
    else:
        nw, nh = round(256 * w / h), 256
    img = img.resize((nw, nh), Image.BILINEAR)
    # center-crop 224
    left, top = (nw - 224) // 2, (nh - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    a = np.asarray(img).astype(np.float32) / 255.0      # [0,1]
    a = (a - 0.5) / 0.5                                  # [-1,1]
    a = np.transpose(a, (2, 0, 1))[None, ...]            # NCHW
    return np.ascontiguousarray(a, dtype=np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def grade_of(fname: str) -> int:
    return int(fname.split("_g")[1][0])


def render(fname, p_ref, decision, gt, correct, dst):
    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(9, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax_img.imshow(Image.open(CAND / fname).convert("RGB"))
    ax_img.axis("off")
    ax_img.set_title(fname, fontsize=8)

    ref_color = "#c0392b" if decision == "REFERABLE" else "#1e8449"
    ax_bar.barh([0], [1.0], color="#e5e7eb", height=0.5, zorder=0)
    ax_bar.barh([0], [p_ref], color=ref_color, height=0.5, zorder=1)
    ax_bar.axvline(THRESHOLD, color="#333", ls="--", lw=1)
    ax_bar.text(THRESHOLD, 0.58, f"thr={THRESHOLD}", fontsize=7, ha="center")
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-1.2, 1.2)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("P(referable)")
    mark = "OK" if correct else "MISS"
    ax_bar.set_title(
        f"{decision}   P(ref)={p_ref:.3f}\nGT grade {gt} "
        f"({'referable' if gt >= 2 else 'not-referable'})  [{mark}]",
        fontsize=10, color=ref_color)
    fig.tight_layout()
    fig.savefig(dst, dpi=110)
    plt.close(fig)


def main():
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    rows = []
    for fname, tag in DEMO:
        x = preprocess(CAND / fname)
        logits = sess.run(None, {iname: x})[0][0]
        probs = softmax(logits)
        assert abs(probs.sum() - 1) < 1e-5, "softmax must sum to 1"
        p_ref = float(probs[1])
        decision = "REFERABLE" if p_ref >= THRESHOLD else "NOT REFERABLE"
        gt = grade_of(fname)
        ref_gt = gt >= 2
        correct = (p_ref >= THRESHOLD) == ref_gt
        stem = f"demo_{tag}_g{gt}_{fname.replace('.png', '')}"
        viz = OUT / f"{stem}_viz.png"
        render(fname, p_ref, decision, gt, correct, viz)
        rows.append(dict(file=fname, tag=tag, gt=gt, ref_gt=ref_gt,
                         logits=[round(float(v), 4) for v in logits],
                         p_referable=round(p_ref, 6), decision=decision,
                         correct=correct, viz=viz.name))
        print(f"{fname:26s} gt={gt} P(ref)={p_ref:.4f} -> {decision:14s} "
              f"[{'OK' if correct else 'MISS'}]  viz={viz.name}")

    out = dict(model="mobilenetv2-dr-referable.onnx", threshold=THRESHOLD,
               labels={"0": "Nrdr (not-referable)", "1": "Rdr (referable)"},
               # aggregate binary metrics from the full 42-image bakeoff eval
               aggregate={"sensitivity": 0.833, "specificity": 0.889,
                          "binary_accuracy": 0.857, "grade0_correct_notref": "6/6",
                          "n": 42, "source": "_eval/mnv2-escvncl/results.json"},
               demo=rows)
    json.dump(out, open(OUT / "results.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'results.json'} and {len(rows)} *_viz.png")


if __name__ == "__main__":
    main()
