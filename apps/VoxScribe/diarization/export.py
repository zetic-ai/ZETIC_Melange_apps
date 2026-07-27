#!/usr/bin/env python3
"""
VoxScribe / diarization — re-runnable export recipe (Stage 0, Explorer)
=======================================================================

WINNER: k2-fsa/sherpa-onnx PRE-EXPORTED pyannote/segmentation-3.0 ONNX.

This is NOT a torch.onnx.export recipe. The whole point of preferring
sherpa-onnx is that the segmentation model is ALREADY exported to ONNX and
published as a plain GitHub release asset (MIT licensed, no Hugging Face
gating, no torch / no pyannote install needed). We download it and then fix
its DYNAMIC axes to STATIC dims so Melange can compile it.

What this script does (all executed successfully by the Explorer on
2026-06-26; see model_selection.md "Export" section):

  1. Download the sherpa-onnx pyannote-segmentation-3.0 tarball (~6.96 MB).
  2. Extract `model.onnx` (FP32, ~5.99 MB). (There is also a model.int8.onnx;
     we deliberately DO NOT use it — Melange handles precision, EXPLORATION
     §"no half precision in the ONNX".)
  3. The published model has DYNAMIC axes:  x:[N,1,T]  y:[N,<symbolic>,7]
     and carries Shape/Slice/If/Gather/ConstantOfShape runtime-shape logic.
     We pin N=1, T=160000 (10 s @ 16 kHz mono) and constant-fold with
     onnx-simplifier, which removes ALL the dynamic-shape machinery.
  4. Strip phantom opset imports (org.pytorch.aten / com.microsoft / etc.)
     that no node actually uses, leaving a single ai.onnx opset-13 domain.
  5. Verify: onnx.checker passes, output is exactly [1,589,7], and the static
     graph is numerically IDENTICAL to the original dynamic graph
     (max abs diff == 0.0).

Output artifact: pyannote_segmentation_static.onnx  (this folder).

Environment used by the Explorer (system python was 3.14 with no numpy; a
uv-managed Python 3.11 venv was used instead):
    uv venv --python 3.11 uvenv
    uv pip install --python ./uvenv/bin/python numpy onnx onnxruntime onnxsim

Re-run:
    python3 export.py
"""

import hashlib
import os
import subprocess
import sys
import tarfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import checker
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
from onnxsim import simplify

# --- Source (GitHub release asset; MIT; NO Hugging Face gating) -------------
SEG_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
# Verified by the Explorer on 2026-06-26:
TARBALL_SHA256 = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"

HERE = os.path.dirname(os.path.abspath(__file__))
TARBALL = os.path.join(HERE, "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EXTRACT_DIR = os.path.join(HERE, "sherpa-onnx-pyannote-segmentation-3-0")
SRC_ONNX = os.path.join(EXTRACT_DIR, "model.onnx")           # FP32 dynamic
OUT_ONNX = os.path.join(HERE, "pyannote_segmentation_static.onnx")
SAMPLE_NPY = os.path.join(HERE, "sample_input.npy")

# --- Fixed static dims for the 10 s window @ 16 kHz mono --------------------
N = 1
SAMPLES = 160_000          # 10.0 s * 16000 Hz
NUM_FRAMES = 589           # model's fixed per-window frame count
NUM_CLASSES = 7            # powerset classes (see model_selection.md)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    if os.path.isfile(SRC_ONNX):
        print(f"[skip] already extracted: {SRC_ONNX}")
        return
    if not os.path.isfile(TARBALL):
        print(f"[download] {SEG_URL}")
        # curl is available everywhere on macOS; wget is not.
        subprocess.run(["curl", "-sL", "-o", TARBALL, SEG_URL], check=True)
    got = sha256(TARBALL)
    print(f"[sha256] {got}")
    if got != TARBALL_SHA256:
        print(f"[warn] checksum mismatch (expected {TARBALL_SHA256}); "
              "upstream may have re-released. Inspect before trusting.")
    with tarfile.open(TARBALL, "r:bz2") as t:
        t.extractall(HERE)
    print(f"[extracted] -> {EXTRACT_DIR}")


def make_static():
    print(f"[load] {SRC_ONNX}")
    m = onnx.load(SRC_ONNX)

    # 1) Pin the symbolic dims by name.
    make_dim_param_fixed(m.graph, "N", N)
    make_dim_param_fixed(m.graph, "T", SAMPLES)

    # 2) Constant-fold Shape/Slice/If/etc. with a concrete input shape.
    m, ok = simplify(m, overwrite_input_shapes={"x": [N, 1, SAMPLES]})
    assert ok, "onnxsim failed to simplify the model"

    # 3) Drop phantom opset imports that no node uses (org.pytorch.aten, etc.).
    used = set()
    for node in m.graph.node:
        used.add(node.domain)
    keep = [o for o in m.opset_import if o.domain in used]
    del m.opset_import[:]
    m.opset_import.extend(keep)

    checker.check_model(m)
    onnx.save(m, OUT_ONNX)
    print(f"[saved] {OUT_ONNX}")
    return m


def verify(m):
    # Declared static shapes
    def shp(t):
        return [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
    assert shp(m.graph.input[0]) == [N, 1, SAMPLES], shp(m.graph.input[0])
    assert shp(m.graph.output[0]) == [N, NUM_FRAMES, NUM_CLASSES], shp(m.graph.output[0])
    assert all(n.op_type not in {"If", "Shape", "Slice", "ConstantOfShape"}
               for n in m.graph.node), "dynamic-shape op still present!"

    # Numerical equivalence to the original dynamic graph
    np.random.seed(0)
    x = np.random.rand(N, 1, SAMPLES).astype(np.float32)
    y_static = ort.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    y_orig = ort.InferenceSession(SRC_ONNX, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    diff = float(np.max(np.abs(y_static - y_orig)))
    print(f"[verify] static output shape = {y_static.shape}, max|static-orig| = {diff}")
    assert y_static.shape == (N, NUM_FRAMES, NUM_CLASSES)
    assert diff == 0.0, diff

    # Sample input for the dashboard (random noise of the right shape/dtype).
    np.save(SAMPLE_NPY, x)
    print(f"[saved] {SAMPLE_NPY}  shape={x.shape} dtype={x.dtype}")


if __name__ == "__main__":
    download()
    model = make_static()
    verify(model)
    print("\nDONE. Drag pyannote_segmentation_static.onnx + sample_input.npy "
          "into Melange (see melange_upload.md).")
    sys.exit(0)
