#!/usr/bin/env python3
"""
Stage-0 export recipe — LiveDocRedact (ZETIC Melange demo).

TWO-MODEL OCR pipeline (live sensitive-document capture / PII auto-redaction):
  1. DETECTOR   — DBNet text-region detector  (finds text-box quads in the frame)
  2. RECOGNIZER — CRNN/SVTR CTC text recognizer (reads chars from each crop)

Both winners come from PaddleOCR (PP-OCRv5 mobile), Apache-2.0 (GTM-clean).
See model_selection.md for the shortlist + rationale.

WHY these are a good Melange fit:
  - Standard conv/BN/CTC ops, mobile-sized (~4.8 MB det, ~8 MB rec).
  - Both export to CLEAN, fully STATIC ONNX (dynamic=False equivalent): we pin the
    input dims to concrete values and constant-fold with onnxslim so NO Shape/Slice/
    If op and NO symbolic axis survives — verified programmatically below.
  - opset 12, FP32 (Melange owns precision — no fp16 baked into the ONNX).

Family recipe note (new for the OCR family):
  - The DETECTOR is taken from PaddlePaddle's PRE-EXPORTED ONNX repo
    (PP-OCRv5_mobile_det_onnx) whose input is dynamic [N,3,H,W]; we pin it to
    [1,3,640,640] and onnxslim-fold.
  - The RECOGNIZER has no pre-exported English ONNX, so we convert the English
    PP-OCRv5 mobile rec Paddle inference model (en_PP-OCRv5_mobile_rec) with
    paddle2onnx (opset 12), then pin its input to the FIXED [1,3,48,320] (variable
    width is a static-shape violation — we resolve it to 320) and onnxslim-fold.
    The English charset (436 chars) also yields a lean [1,40,438] CTC head vs the
    18,385-class multilingual head — far cheaper per-crop decode for a Latin
    ID/passport/medical demo.

Environment used by the Explorer (system python 3.14 is broken — do NOT use it):
    uv venv --python 3.12 .venv-livedocredact
    uv pip install --python .venv-livedocredact/bin/python \
        huggingface_hub numpy onnx onnxruntime paddlepaddle paddle2onnx onnxslim
Re-run:
    .venv-livedocredact/bin/python export.py

Outputs (this folder):
    doc_text_detector.onnx        + detector_sample_input.npy
    doc_text_recognizer.onnx      + recognizer_sample_input.npy
    en_dict.txt                   (recognizer CTC character dictionary — Dart asset)
"""
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import onnxslim
import paddle2onnx
import yaml
from huggingface_hub import hf_hub_download
from onnx import version_converter

HERE = Path(__file__).resolve().parent

OPSET = 12   # known-good for Melange (PyroGuard); recognizer exports at 12,
             # detector's pre-exported ONNX is opset 11 -> we upgrade it to 12.

# --- Winners -----------------------------------------------------------------
DET_REPO = "PaddlePaddle/PP-OCRv5_mobile_det_onnx"    # pre-exported ONNX, Apache-2.0
REC_REPO = "PaddlePaddle/en_PP-OCRv5_mobile_rec"      # Paddle inference model, Apache-2.0

# --- Fixed static input shapes (STATIC SHAPES OR BUST) -----------------------
DET_SHAPE = [1, 3, 640, 640]   # NCHW; DBNet is fully-conv, dims must be /32
REC_SHAPE = [1, 3, 48, 320]    # NCHW; PP-OCR rec fixed H=48, FIXED W=320

DET_ONNX = HERE / "doc_text_detector.onnx"
REC_ONNX = HERE / "doc_text_recognizer.onnx"


def _dims(t):
    return [d.dim_value if d.dim_value else (d.dim_param or "?")
            for d in t.type.tensor_type.shape.dim]


def pin_and_fold(model: onnx.ModelProto, shape) -> onnx.ModelProto:
    """Overwrite the single input's dims to concrete values (name-agnostic),
    then constant-fold with onnxslim so every downstream Shape/Reshape resolves
    to a concrete static output. Returns the folded model."""
    dim = model.graph.input[0].type.tensor_type.shape.dim
    assert len(dim) == len(shape), f"rank mismatch {len(dim)} vs {len(shape)}"
    for d, v in zip(dim, shape):
        d.ClearField("dim_param")
        d.dim_value = int(v)
    return onnxslim.slim(model)


def assert_static_and_run(path: Path, shape):
    """Hard gate: NO symbolic axis on any input/output, NO Shape op left, and the
    graph actually runs in ORT at the declared shape. Prints the real I/O."""
    m = onnx.load(str(path))
    ins = [(i.name, _dims(i)) for i in m.graph.input]
    outs = [(o.name, _dims(o)) for o in m.graph.output]
    bad = [d for _, ds in ins + outs for d in ds if not isinstance(d, int)]
    assert not bad, f"DYNAMIC AXES REMAIN in {path.name}: {bad}"
    leftover = [n.op_type for n in m.graph.node
                if n.op_type in {"Shape", "If", "Loop", "NonMaxSuppression"}]
    assert not leftover, f"dynamic-shape ops remain in {path.name}: {set(leftover)}"
    x = np.random.rand(*shape).astype(np.float32)
    y = ort.InferenceSession(str(path),
                             providers=["CPUExecutionProvider"]).run(
                                 None, {m.graph.input[0].name: x})[0]
    print(f"  [{path.name}] IN {ins}  OUT {outs}")
    print(f"  [{path.name}] static OK (no dynamic axes, no Shape ops); "
          f"ORT run -> {tuple(y.shape)}")
    return outs


def export_detector():
    print("\n== DETECTOR: PP-OCRv5 mobile DBNet ==")
    src = hf_hub_download(DET_REPO, "inference.onnx")
    m = onnx.load(src)
    m = pin_and_fold(m, DET_SHAPE)
    if m.opset_import[0].version != OPSET:      # upgrade opset 11 -> 12
        m = version_converter.convert_version(m, OPSET)
        m = onnxslim.slim(m)                     # re-fold after conversion
    onnx.save(m, str(DET_ONNX))
    print(f"  saved -> {DET_ONNX}")
    assert_static_and_run(DET_ONNX, DET_SHAPE)
    np.save(HERE / "detector_sample_input.npy",
            np.random.rand(*DET_SHAPE).astype(np.float32))
    print("  saved -> detector_sample_input.npy "
          f"(float32 {tuple(DET_SHAPE)})")


def export_recognizer():
    print("\n== RECOGNIZER: en PP-OCRv5 mobile CTC ==")
    json_path = hf_hub_download(REC_REPO, "inference.json")
    hf_hub_download(REC_REPO, "inference.pdiparams")  # sibling of inference.json
    yml_path = hf_hub_download(REC_REPO, "inference.yml")
    model_dir = os.path.dirname(json_path)

    # 1. paddle2onnx -> dynamic-axis ONNX, opset 12 (Melange owns precision -> fp16 off)
    dyn = HERE / "_rec_dynamic.onnx"
    paddle2onnx.export(
        model_filename=os.path.join(model_dir, "inference.json"),
        params_filename=os.path.join(model_dir, "inference.pdiparams"),
        save_file=str(dyn),
        opset_version=12,
        auto_upgrade_opset=True,
        enable_onnx_checker=True,
        export_fp16_model=False,
    )
    # 2. pin FIXED input [1,3,48,320] + constant-fold -> static [1,40,438]
    m = pin_and_fold(onnx.load(str(dyn)), REC_SHAPE)
    onnx.save(m, str(REC_ONNX))
    dyn.unlink(missing_ok=True)
    print(f"  saved -> {REC_ONNX}")
    outs = assert_static_and_run(REC_ONNX, REC_SHAPE)
    np.save(HERE / "recognizer_sample_input.npy",
            np.random.rand(*REC_SHAPE).astype(np.float32))
    print("  saved -> recognizer_sample_input.npy "
          f"(float32 {tuple(REC_SHAPE)})")

    # 3. Emit the CTC character dictionary (Dart NEEDS this to decode).
    #    PP-OCR CTC label list = ['blank'(0)] + character_dict + [' '].
    #    We write only the 436 dict chars (one per line, index i -> label i+1);
    #    index 0 is the CTC blank, the final class (index 437) is space ' '.
    chars = yaml.safe_load(open(yml_path))["PostProcess"]["character_dict"]
    (HERE / "en_dict.txt").write_text(
        "\n".join(chars) + "\n", encoding="utf-8")
    out_classes = outs[0][1][-1]
    print(f"  saved -> en_dict.txt ({len(chars)} chars); "
          f"CTC classes = 1(blank) + {len(chars)}(dict) + 1(space) = "
          f"{len(chars) + 2} (ONNX head = {out_classes})")
    assert len(chars) + 2 == out_classes, "charset/head size mismatch!"


if __name__ == "__main__":
    export_detector()
    export_recognizer()
    print("\nDONE. See melange_upload.md for the two GATE-0 dashboard uploads.")
