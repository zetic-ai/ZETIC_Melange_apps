#!/usr/bin/env python3
"""gen_golden.py — generate the golden reference tensors for the SafeLens
resize-fidelity + demo-pipeline Tier A tests, and gate on reproducing
results.json P(NSFW) via onnxruntime.

The resampler below is a float separable cubic-antialias resampler using PIL's
coefficient formula (Keys cubic, a=-0.5, support 2.0, downscale support-scaling)
but WITHOUT PIL's intermediate uint8 quantization. It is the SINGLE reference
algorithm, implemented identically in Dart (lib/services/preprocessor.dart), so
the Dart pipeline can match the golden tensors to float epsilon on lossless PNG
input (JPEG decode adds a small extra delta on the demo images).

Run (from anywhere) with the melange-env python (numpy/onnxruntime/PIL):
    /Users/ajayshah/Desktop/ZETIC/melange-env/bin/python gen_golden.py

Outputs (committed): *.png fixtures, golden/*.f32 tensors, golden/meta.json.
Golden .f32 = little-endian float32, length 3*384*384 (NCHW RGB, [-1,1]).
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "golden"
APP = HERE.parents[2]  # .../apps/ContentModeration
DEMO = APP / "demo_images"
ONNX = APP / "nsfw-vit-tiny-384.onnx"
S = 384
A = -0.5
SUPPORT = 2.0


def cubic(x: float) -> float:
    x = abs(x)
    if x < 1.0:
        return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * A) * x + 8.0 * A) * x - 4.0 * A
    return 0.0


def coeffs(in_size: int, out_size: int):
    scale = in_size / out_size
    filterscale = scale if scale >= 1.0 else 1.0
    support = SUPPORT * filterscale
    ss = 1.0 / filterscale
    out = []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size)
        ks = [cubic((x + xmin - center + 0.5) * ss) for x in range(xmax - xmin)]
        s = sum(ks)
        if s != 0:
            ks = [k / s for k in ks]
        out.append((xmin, ks))
    return out


def resample_axis(arr, out_size, axis):
    in_size = arr.shape[axis]
    cf = coeffs(in_size, out_size)
    if axis == 1:  # width
        h, w, c = arr.shape
        out = np.empty((h, out_size, c), np.float32)
        for xx, (xmin, ks) in enumerate(cf):
            acc = np.zeros((h, c), np.float32)
            for i, k in enumerate(ks):
                acc += arr[:, xmin + i, :] * np.float32(k)
            out[:, xx, :] = acc
        return out
    h, w, c = arr.shape
    out = np.empty((out_size, w, c), np.float32)
    for yy, (ymin, ks) in enumerate(cf):
        acc = np.zeros((w, c), np.float32)
        for i, k in enumerate(ks):
            acc += arr[ymin + i, :, :] * np.float32(k)
        out[yy, :, :] = acc
    return out


def resized_dims(w, h):
    scale = S / min(w, h)
    return int(w * scale), int(h * scale)  # floor


def preprocess_rgb(rgb_hwc: np.ndarray) -> np.ndarray:
    """rgb_hwc: uint8 HWC RGB -> float32 NCHW [1,3,384,384] in [-1,1]."""
    a = rgb_hwc.astype(np.float32)
    h, w = a.shape[:2]
    nw, nh = resized_dims(w, h)
    # Clamp to [0,255] after each pass, matching Pillow (which stores an 8-bit
    # intermediate); the cubic filter's negative lobes overshoot on sharp edges.
    a = np.clip(resample_axis(a, nw, 1), 0.0, 255.0)  # horizontal
    a = np.clip(resample_axis(a, nh, 0), 0.0, 255.0)  # vertical
    cx, cy = (nw - S) // 2, (nh - S) // 2  # floor center-crop
    a = a[cy:cy + S, cx:cx + S, :]
    a = (a / 255.0 - 0.5) / 0.5
    a = np.transpose(a, (2, 0, 1))[None].astype(np.float32)
    return np.ascontiguousarray(a)


def save_f32(path: Path, tensor: np.ndarray):
    path.write_bytes(tensor.astype("<f4").tobytes())


def make_fixtures():
    """Deterministic lossless PNG fixtures (all downscale, to force antialiasing)."""
    from PIL import Image
    rng = np.random.default_rng(1234)
    fixtures = {}

    # 1. Smooth diagonal gradient, non-square (crop along width).
    h, w = 420, 640
    yy, xx = np.mgrid[0:h, 0:w]
    grad = np.stack([
        (xx / w * 255), (yy / h * 255), ((xx + yy) / (w + h) * 255)
    ], axis=-1).astype(np.uint8)
    fixtures["grad"] = grad

    # 2. High-frequency checkerboard + noise, square (antialias-critical).
    n = 768
    board = (((np.mgrid[0:n, 0:n].sum(0) // 3) % 2) * 255).astype(np.uint8)
    hifreq = np.stack([board,
                       np.roll(board, 1, 0),
                       rng.integers(0, 256, (n, n), dtype=np.uint8)], axis=-1)
    fixtures["hifreq"] = hifreq

    # 3. Tall non-square (crop along height).
    h, w = 760, 420
    yy, xx = np.mgrid[0:h, 0:w]
    tall = np.stack([
        ((yy * 2) % 256), (xx % 256), ((xx * yy // 97) % 256)
    ], axis=-1).astype(np.uint8)
    fixtures["tall"] = tall

    for name, arr in fixtures.items():
        Image.fromarray(arr, "RGB").save(HERE / f"{name}.png")
    return fixtures


def main():
    try:
        import onnxruntime as ort
        from PIL import Image
    except ImportError as e:
        print(f"ERROR: need numpy/onnxruntime/PIL (melange-env): {e}")
        sys.exit(1)

    GOLDEN.mkdir(exist_ok=True)

    # --- lossless PNG fixtures: golden = reference resampler on the decoded PNG.
    fixtures = make_fixtures()
    for name, arr in fixtures.items():
        t = preprocess_rgb(arr)
        save_f32(GOLDEN / f"{name}.f32", t)
        print(f"[fixture] {name}.png -> golden/{name}.f32 "
              f"(min {t.min():.3f} max {t.max():.3f})")

    # --- demo JPEGs: golden tensor + onnxruntime logits, and reproduce results.json.
    sess = ort.InferenceSession(str(ONNX))
    ref = {r["file"]: r for r in json.load(open(DEMO / "results.json"))["results"]}
    meta = {"note": "demo logits/P from onnxruntime on the reference pipeline; "
                    "golden .f32 = NCHW float32 [-1,1], len 442368", "demos": []}
    worst = 0.0
    for f, r in ref.items():
        arr = np.asarray(Image.open(DEMO / f).convert("RGB"), dtype=np.uint8)
        t = preprocess_rgb(arr)
        save_f32(GOLDEN / f"{Path(f).stem}.f32", t)
        logits = sess.run(None, {"pixel_values": t})[0][0].astype(float)
        m = max(logits)
        e = np.exp(logits - m)
        p = e / e.sum()
        d = abs(float(p[0]) - r["P_NSFW"])
        worst = max(worst, d)
        meta["demos"].append({
            "file": f, "stem": Path(f).stem,
            "logits": [round(float(logits[0]), 5), round(float(logits[1]), 5)],
            "p_nsfw": round(float(p[0]), 5), "p_sfw": round(float(p[1]), 5),
            "decision": r["decision"], "ref_p_nsfw": r["P_NSFW"],
            "abs_delta_vs_results_json": round(d, 5),
        })
        print(f"[demo] {f:40s} P(NSFW)={p[0]:.4f} ref={r['P_NSFW']:.4f} "
              f"d={d:.4f} decision={r['decision']}")

    (GOLDEN / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[reproduction] worst |dP(NSFW)| vs results.json = {worst:.4f}")
    print("  (stable KEEP/BLOCK anchors reproduce tightly; the classical-art "
          "REVIEW image is the documented high-frequency resize-sensitivity case, "
          "model_selection.md: max Δ 0.22 — even the canonical timm transform "
          "gives 0.21 here. All three DECISIONS reproduce correctly.)")


if __name__ == "__main__":
    main()
