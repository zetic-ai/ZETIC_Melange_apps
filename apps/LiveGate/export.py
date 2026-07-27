#!/usr/bin/env python3
"""
Stage-0 export recipe — LiveGate (ZETIC Melange demo).

TWO-MODEL on-device KYC gate (live-selfie liveness + 1:1 face match):
  1. PAD   — MiniFASNet-V2 (2.7_80x80) silent face anti-spoofing classifier
             (is this a REAL live face, or a printed-photo / screen-replay spoof?)
  2. FACE  — SFace (OpenCV Zoo) MobileFaceNet-style ArcFace embedding
             (512-input 128-d face embedding for cosine 1:1 verification)

Both winners were VALIDATION-GATED (see model_selection.md for the measured numbers):
  - FACE  SFace: 94.8% LFW 1:1 verification accuracy (crude alignment; higher with
          real 5-point alignment on-device), same-cos 0.505 vs diff-cos 0.166.
  - PAD   MiniFASNet-V2: 80.8% live/spoof accuracy on a 60-live/60-spoof CelebA-Spoof
          subset with the model's exact 2.7x crop geometry.

WHY these are a good Melange fit:
  - Standard conv/BN/PRelu/Gemm ops, mobile-sized (PAD ~1.7 MB, FACE ~38.5 MB).
  - Both resolve to CLEAN, fully STATIC ONNX (batch pinned to 1, no dynamic axis,
    no Shape/If/Loop/Gather left after constant-folding) — verified below.
  - opset 12, FP32 (Melange owns precision — no fp16 baked into the ONNX).

LICENSE (both GTM-clean / permissive):
  - PAD  Apache-2.0 (minivision-ai Silent-Face-Anti-Spoofing, via garciafido mirror).
  - FACE Apache-2.0 (OpenCV Zoo SFace).
  NOTE: the popular InsightFace w600k_mbf (buffalo_s) was REJECTED for the FACE slot —
  it is non-commercial-research-only, which sinks a GTM demo. SFace is Apache-2.0.

Family recipe note (new for the FACE-BIOMETRICS family):
  - PAD is taken as a pre-exported ONNX (opset 11, dynamic batch); we pin batch=1,
    upgrade to opset 12, and onnxslim-fold so Shape/Gather vanish.
  - FACE (SFace) is a pre-exported ONNX (MXNet->ONNX style) whose graph.input lists
    ALL 174 weight tensors alongside the real image input `data`; we strip the
    initializers from graph.input (leaving only `data`), upgrade to opset 12, and
    onnxslim-fold. Normalization ((x-127.5)/128) is BAKED INTO the graph, so the
    app feeds raw 0..255 BGR pixels.

*** HARD-WON EMPIRICAL FINDINGS (the model cards are WRONG — trust the ONNX) ***
  PAD input range: this ONNX discriminates ONLY on raw BGR [0,255]. Feeding pixel/255
    ([0,1], as the model card claims) SATURATES it to a constant class-2 output (dead).
    Validation caught this; see model_selection.md.
  PAD live class: LIVE = softmax[class 1] (MiniVision test.py convention), NOT class 0
    as the garciafido card's "[live,print,replay]" ordering claims. Empirically class 1
    is the live indicator; class 0/2 are the two spoof types.

Environment used by the Explorer (venv at /Users/ajayshah/Desktop/ZETIC/melange-env):
    huggingface_hub, numpy, onnx, onnxruntime, onnxslim  (all already installed)
Re-run:
    /Users/ajayshah/Desktop/ZETIC/melange-env/bin/python export.py

Outputs (this folder):
    pad_minifasnet_v2.onnx     + sample_input_pad.npy    (float32 [1,3,80,80])
    face_sface.onnx            + sample_input_face.npy    (float32 [1,3,112,112])
"""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
from huggingface_hub import hf_hub_download
from onnx import version_converter

HERE = Path(__file__).resolve().parent
OPSET = 12  # known-good for Melange (PyroGuard). Both sources are opset 11 -> upgrade.

# --- Winners ----------------------------------------------------------------
PAD_REPO  = "garciafido/minifasnet-v2-anti-spoofing-onnx"   # Apache-2.0, pre-exported ONNX
PAD_FILE  = "minifasnet_v2.onnx"
FACE_REPO = "opencv/face_recognition_sface"                 # Apache-2.0, pre-exported ONNX
FACE_FILE = "face_recognition_sface_2021dec.onnx"

# --- Fixed static input shapes (STATIC SHAPES OR BUST) ----------------------
PAD_SHAPE  = [1, 3, 80, 80]     # NCHW, BGR, RANGE [0,255] (NOT /255 — see header)
FACE_SHAPE = [1, 3, 112, 112]   # NCHW, BGR, RANGE [0,255] (in-graph (x-127.5)/128)

PAD_ONNX  = HERE / "pad_minifasnet_v2.onnx"
FACE_ONNX = HERE / "face_sface.onnx"


def _dims(t):
    return [d.dim_value if d.dim_value else (d.dim_param or "?")
            for d in t.type.tensor_type.shape.dim]


def assert_static_and_run(path: Path, shape):
    """Hard gate: NO symbolic axis on any input/output, NO Shape/If/Loop/Gather op
    left, and the graph actually runs in ORT at the declared shape."""
    m = onnx.load(str(path))
    ins = [(i.name, _dims(i)) for i in m.graph.input]
    outs = [(o.name, _dims(o)) for o in m.graph.output]
    bad = [d for _, ds in ins + outs for d in ds if not isinstance(d, int)]
    assert not bad, f"DYNAMIC AXES REMAIN in {path.name}: {bad}"
    leftover = [n.op_type for n in m.graph.node
                if n.op_type in {"Shape", "If", "Loop", "NonMaxSuppression", "Gather"}]
    assert not leftover, f"dynamic-shape ops remain in {path.name}: {set(leftover)}"
    assert len(m.graph.input) == 1, \
        f"expected exactly 1 graph input, got {len(m.graph.input)}: {[i.name for i in m.graph.input]}"
    x = np.random.rand(*shape).astype(np.float32)
    y = ort.InferenceSession(str(path),
                             providers=["CPUExecutionProvider"]).run(
                                 None, {m.graph.input[0].name: x})[0]
    print(f"  [{path.name}] IN {ins}  OUT {outs}")
    print(f"  [{path.name}] static OK (1 input, no dynamic axes, no Shape/Gather); "
          f"ORT run -> {tuple(y.shape)}")


def export_pad():
    print("\n== PAD: MiniFASNet-V2 (2.7_80x80) silent anti-spoofing ==")
    src = hf_hub_download(PAD_REPO, PAD_FILE)
    m = onnx.load(src)
    # pin batch axis (source is ['batch',3,80,80]) to 1
    d0 = m.graph.input[0].type.tensor_type.shape.dim[0]
    d0.ClearField("dim_param")
    d0.dim_value = 1
    m = onnxslim.slim(m)                       # fold Shape/Gather -> constant Reshape
    if m.opset_import[0].version != OPSET:
        m = version_converter.convert_version(m, OPSET)
        m = onnxslim.slim(m)
    onnx.save(m, str(PAD_ONNX))
    print(f"  saved -> {PAD_ONNX} ({PAD_ONNX.stat().st_size/1e6:.2f} MB)")
    assert_static_and_run(PAD_ONNX, PAD_SHAPE)
    np.save(HERE / "sample_input_pad.npy",
            np.random.rand(*PAD_SHAPE).astype(np.float32))
    print(f"  saved -> sample_input_pad.npy (float32 {tuple(PAD_SHAPE)})")


def export_face():
    print("\n== FACE: SFace (OpenCV Zoo) ArcFace embedding ==")
    src = hf_hub_download(FACE_REPO, FACE_FILE)
    m = onnx.load(src)
    # SFace lists all 174 weight tensors in graph.input (old MXNet->ONNX style).
    # Keep ONLY the real image input `data`; the weights stay as initializers.
    init_names = {i.name for i in m.graph.initializer}
    keep = [i for i in m.graph.input if i.name not in init_names]
    assert [i.name for i in keep] == ["data"], f"unexpected real inputs: {[i.name for i in keep]}"
    del m.graph.input[:]
    m.graph.input.extend(keep)
    m = onnxslim.slim(m)
    if m.opset_import[0].version != OPSET:
        m = version_converter.convert_version(m, OPSET)
        m = onnxslim.slim(m)
    onnx.save(m, str(FACE_ONNX))
    print(f"  saved -> {FACE_ONNX} ({FACE_ONNX.stat().st_size/1e6:.2f} MB)")
    assert_static_and_run(FACE_ONNX, FACE_SHAPE)
    np.save(HERE / "sample_input_face.npy",
            np.random.rand(*FACE_SHAPE).astype(np.float32))
    print(f"  saved -> sample_input_face.npy (float32 {tuple(FACE_SHAPE)})")


if __name__ == "__main__":
    export_pad()
    export_face()
    print("\nDONE. See melange_upload.md for the two GATE-0 dashboard uploads.")
