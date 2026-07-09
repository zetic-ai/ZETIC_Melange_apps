"""
validate_demo.py — Re-runnable demo validation + visualization for RetinaDRGrade.

For each demo fundus image in demo_images/, runs the SAME preprocessing the ViT model
was trained with (224 bilinear, RGB, /255, mean/std [0.5,0.5,0.5], NCHW), runs the
exported ONNX (or the torch model), applies softmax + argmax, and renders a *_viz.png:
  [ fundus image ]  |  [ 5-grade softmax bar, GT vs predicted grade annotated ]

Grades: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative.  Referable = grade >= 2.
id2label is identity, so argmax index == canonical grade directly.

Run:
  pip install onnxruntime pillow numpy matplotlib
  python validate_demo.py
"""
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "vit-base-dr-grade.onnx")
DEMO = os.path.join(HERE, "demo_images")

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Ground-truth grade parsed from the filename convention <DATASET>_g<GRADE>_<hash>.png
def gt_from_name(fn):
    import re
    m = re.search(r"_g(\d)_", fn)
    return int(m.group(1)) if m else None


def preprocess(path):
    """224 bilinear resize, RGB, /255, normalize [0.5]/[0.5], NCHW float32 [1,3,224,224]."""
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, [0,1]
    arr = (arr - MEAN) / STD                                  # normalize -> [-1,1]
    arr = np.transpose(arr, (2, 0, 1))                        # CHW
    return arr[None, :, :, :].astype(np.float32), img         # NCHW + display img


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main():
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    files = sorted(f for f in os.listdir(DEMO)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_viz" not in f)
    n_exact = 0
    ref_tp = ref_fn = ref_tn = ref_fp = 0
    for f in files:
        path = os.path.join(DEMO, f)
        x, disp = preprocess(path)
        logits = sess.run(["logits"], {"pixel_values": x})[0][0]
        probs = softmax(logits)
        pred = int(np.argmax(probs))
        gt = gt_from_name(f)
        exact = (gt == pred)
        n_exact += int(exact)
        ref_gt, ref_pred = gt >= 2, pred >= 2
        ref_tp += int(ref_gt and ref_pred); ref_fn += int(ref_gt and not ref_pred)
        ref_tn += int(not ref_gt and not ref_pred); ref_fp += int(not ref_gt and ref_pred)
        print(f"{f}: GT={gt} ({GRADE_NAMES[gt]})  PRED={pred} ({GRADE_NAMES[pred]})  "
              f"p={probs.round(4).tolist()}  referable(pred)={ref_pred}")

        # ---- viz ----
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
        axL.imshow(disp); axL.axis("off")
        axL.set_title(f, fontsize=9)
        colors = ["#3a7d44" if i < 2 else "#c1432b" for i in range(5)]  # green<2, red>=2
        bars = axR.bar(range(5), probs, color=colors)
        bars[pred].set_edgecolor("black"); bars[pred].set_linewidth(2.5)
        axR.set_xticks(range(5))
        axR.set_xticklabels([f"{i}\n{GRADE_NAMES[i]}" for i in range(5)], fontsize=8)
        axR.set_ylim(0, 1); axR.set_ylabel("softmax probability")
        for i, p in enumerate(probs):
            axR.text(i, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ref_txt = "REFERABLE (>=2)" if pred >= 2 else "not referable (<2)"
        ok = "OK" if exact else "MISS"
        axR.set_title(f"GT grade {gt} ({GRADE_NAMES[gt]})  |  pred {pred} "
                      f"({GRADE_NAMES[pred]}) [{ok}]\n{ref_txt}", fontsize=10)
        fig.tight_layout()
        out = os.path.join(DEMO, os.path.splitext(f)[0] + "_viz.png")
        fig.savefig(out, dpi=110); plt.close(fig)
        print("  wrote", os.path.basename(out))

    N = len(files)
    sens = ref_tp / (ref_tp + ref_fn) if (ref_tp + ref_fn) else None
    spec = ref_tn / (ref_tn + ref_fp) if (ref_tn + ref_fp) else None
    print(f"\nDemo set: N={N}  exact-grade correct={n_exact}/{N}")
    print(f"Referable(>=2): sens={sens}  spec={spec}  (tp={ref_tp} fn={ref_fn} tn={ref_tn} fp={ref_fp})")


if __name__ == "__main__":
    main()
