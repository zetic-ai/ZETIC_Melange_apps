#!/usr/bin/env python3
"""
Stage-0 export recipe — ReceiptScanner (ZETIC Melange OCR demo).

Family: OCR (PaddleOCR PP-OCRv5 mobile). This is the FIRST OCR-family recipe, so it
also establishes the reusable PP-OCR export path (paddle2onnx -> static-shape fix).

Pipeline shape: TWO Melange models (detector + recognizer). A single recognizer cannot
localize the many text lines on a full receipt (recognizers read ONE pre-cropped line),
so producing "text + boxes" for a whole receipt genuinely needs detect-then-recognize.
See model_selection.md for the justified fallback rationale. The structured extraction
(line-items / totals / merchant / date) is pure-Dart post-processing on top of the
model's text+box output — NOT a model.

Winner models (both Apache-2.0 — GTM-clean):
  DET: PaddlePaddle/PP-OCRv5_mobile_det   (DB text detector, ~4.5 MB ONNX)
  REC: PaddlePaddle/en_PP-OCRv5_mobile_rec (SVTR/CTC English recognizer, ~7.5 MB ONNX)
       (English/Latin recognizer chosen over the ready 18385-class multilingual ONNX:
        436-char receipt-friendly charset incl. currency symbols; smaller + trivial
        CTC decode. See model_selection.md.)

What this script does, re-runnably:
  1. Download the two PP-OCRv5 Paddle inference models from the Hugging Face Hub.
  2. paddle2onnx -> ONNX at opset 12 (Melange owns precision; no fp16 here).
  3. onnxslim with FIXED input shapes -> STATIC-shape ONNX (folds all dynamic
     Shape/Reshape/Slice nodes into constants).
         DET input  fixed to [1,3,960,640]  (portrait, receipt-friendly, /32)
         REC input  fixed to [1,3,48,320]   (PP-OCR canonical rec image_shape)
  4. Emit one sample_input.npy per model (np.random.rand, correct shape/dtype only).
  5. Emit the recognizer decode table (rec_charset.json / rec_dict.txt).
  6. Programmatically VERIFY both ONNX have NO dynamic axes and the expected I/O, and
     run a real onnxruntime forward pass to confirm output shapes.

Toolchain (do NOT use broken system python 3.14 — use an isolated 3.12 venv):
  uv venv --python 3.12 .venv-receiptscanner
  uv pip install --python .venv-receiptscanner/bin/python \
      huggingface_hub onnx onnxruntime onnxslim numpy pyyaml \
      paddlepaddle paddle2onnx
  .venv-receiptscanner/bin/python export.py

Note: paddle2onnx 2.x imports `paddle`, so paddlepaddle must be installed even though
we only convert (no training/inference in Paddle). onnxsim segfaulted on this graph on
macOS/arm64; onnxslim is the reliable static-shape simplifier here.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
BIN = Path(sys.executable).parent          # venv bin dir (paddle2onnx / onnxslim scripts)

MODELS = {
    "det": {
        "repo": "PaddlePaddle/PP-OCRv5_mobile_det",
        "onnx": "ppocrv5_mobile_det.onnx",
        "shape": (1, 3, 960, 640),          # N,C,H,W  (fixed static)
        "sample": "ppocrv5_mobile_det_sample_input.npy",
    },
    "rec": {
        "repo": "PaddlePaddle/en_PP-OCRv5_mobile_rec",
        "onnx": "en_ppocrv5_mobile_rec.onnx",
        "shape": (1, 3, 48, 320),           # N,C,H,W  (fixed static; W fixes CTC width)
        "sample": "en_ppocrv5_mobile_rec_sample_input.npy",
    },
}
OPSET = 12


def run(cmd):
    print("[cmd]", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def export_one(key, cfg):
    repo = cfg["repo"]
    # 1. Download the Paddle 3.0 inference model (JSON graph + params + config yml)
    mdir = Path(hf_hub_download(repo, "inference.json")).parent
    hf_hub_download(repo, "inference.pdiparams")
    yml_path = hf_hub_download(repo, "inference.yml")
    print(f"[download] {repo} -> {mdir}")

    raw = HERE / f"_{key}_raw.onnx"
    dst = HERE / cfg["onnx"]

    # 2. paddle2onnx -> opset 12  (console script; `python -m paddle2onnx` is unsupported)
    run([str(BIN / "paddle2onnx"),
         "--model_dir", str(mdir),
         "--model_filename", "inference.json",
         "--params_filename", "inference.pdiparams",
         "--save_file", str(raw),
         "--opset_version", str(OPSET)])

    # 3. onnxslim with FIXED input shape -> static, constant-folded ONNX
    n, c, h, w = cfg["shape"]
    run([str(BIN / "onnxslim"), str(raw), str(dst),
         "--input-shapes", f"x:{n},{c},{h},{w}"])
    raw.unlink(missing_ok=True)

    # 4. sample_input.npy — random noise, correct shape/dtype only
    np.save(HERE / cfg["sample"], np.random.rand(*cfg["shape"]).astype(np.float32))

    return yml_path


def emit_charset(rec_yml):
    """Recognizer decode table in CTC index order: 0=blank, 1..N=dict, N+1=space."""
    chars = yaml.safe_load(open(rec_yml))["PostProcess"]["character_dict"]
    table = [""] + list(chars) + [" "]           # blank + dict + space
    json.dump(table, open(HERE / "rec_charset.json", "w"),
              ensure_ascii=False, indent=0)
    open(HERE / "rec_dict.txt", "w").write("\n".join(chars) + "\n")
    print(f"[charset] {len(table)} classes (blank=0, dict=1..{len(chars)}, "
          f"space={len(chars) + 1})")


def verify(cfg):
    dst = HERE / cfg["onnx"]
    m = onnx.load(str(dst))
    onnx.checker.check_model(m)
    bad = []
    for group in (m.graph.input, m.graph.output, m.graph.value_info):
        for v in group:
            for d in v.type.tensor_type.shape.dim:
                if d.dim_param or not d.HasField("dim_value"):
                    bad.append((v.name, d.dim_param))
    assert not bad, f"DYNAMIC AXES in {cfg['onnx']}: {bad}"
    assert all(init.data_type != onnx.TensorProto.FLOAT16
               for init in m.graph.initializer), "fp16 initializer present"

    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    out = sess.run(None, {iname: np.random.rand(*cfg["shape"]).astype(np.float32)})[0]
    print(f"[verify] {cfg['onnx']}: static OK, opset {m.opset_import[0].version}, "
          f"in {iname}{tuple(cfg['shape'])} -> out {out.shape} {out.dtype}")


def main():
    rec_yml = None
    for key, cfg in MODELS.items():
        y = export_one(key, cfg)
        if key == "rec":
            rec_yml = y
    emit_charset(rec_yml)
    for cfg in MODELS.values():
        verify(cfg)
    print("\n[done] artifacts written to", HERE)


if __name__ == "__main__":
    main()
