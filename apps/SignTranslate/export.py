#!/usr/bin/env python3
"""
Stage-0 export recipe — SignTranslate (ZETIC Melange demo, OCR family).

App:   SignTranslate — LIVE camera offline scene-text reading for a traveler.
Shape: TWO-MODEL pipeline (both required, both registered as separate Melange models):
         1. text DETECTOR   — PP-OCRv5 mobile det (DBNet, MobileNetV3) -> ajayshah/SceneTextDetector
         2. text RECOGNIZER — latin PP-OCRv5 mobile rec (SVTR-LCNet, CTC) -> ajayshah/SceneTextRecognizer
       Text-region grouping + crop-feed orchestration between the two models is Dart
       (worker's job, post GATE 0). The OPTIONAL translate step is Dart / out-of-scope
       here — NO translation model is exported.

Family: OCR / PaddleOCR (PP-OCR). This is the first OCR-family export recipe, so it is
        recorded here for reuse. Both models come from ONE toolchain (paddle2onnx),
        matching the "one recipe per architecture family" rule.

Why scene-text (defended fully in model_selection.md):
  - DET  PP-OCRv5 mobile det is a DBNet scene-text detector (trained on scene + doc text;
         handles arbitrary-orientation text regions — required for angled street signs).
  - REC  latin PP-OCRv5 mobile rec is an SVTR-LCNet recognizer. SVTR is a *scene-text*
         recognition architecture (evaluated on IC13/IC15/SVT/CUTE); PP-OCR trains it on
         a large synthetic-scene + real corpus. It is CTC (NOT autoregressive), so it
         exports to a clean STATIC-shape opset-12 ONNX — avoiding the PARSeq/ABINet
         autoregressive-op export trap called out in the assignment. Latin charset covers
         the bulk of the offline-travel case (French/Spanish/German/Italian/Portuguese...).

What this script does, re-runnably:
  1. Download both winner PaddleOCR inference models from the Hugging Face Hub.
  2. paddle2onnx each -> raw ONNX (dynamic axes, as Paddle exports them).
  3. onnxslim with a FIXED input shape -> STATIC-shape ONNX (opset 12, FP32).
       DET input fixed to [1,3,736,736]  (H,W divisible by 32 for DBNet; 736 = PP-OCR
       TRT optimum shape; letterbox the live frame to this square in Dart).
       REC input fixed to [1,3,48,320]   (fixes the variable text width -> static shapes).
  4. Copy the two <model>.onnx into this app folder.
  5. Emit <model>_sample_input.npy = np.random.rand(shape).float32 (shape/dtype only).
  6. Extract the recognizer CTC character dictionary -> latin_charset.txt (Dart asset).
  7. Programmatically ASSERT there are NO dynamic axes on either exported ONNX.
  8. Print the ACTUAL input/output tensor shapes read back from each exported ONNX.

Toolchain (proven working — do NOT use the broken system python 3.14):
  uv venv --python 3.12 .venv-signtranslate
  uv pip install --python .venv-signtranslate/bin/python \
      paddlepaddle paddleocr paddle2onnx onnx onnxslim onnxruntime numpy huggingface_hub pyyaml
  .venv-signtranslate/bin/python export.py
Versions proven: paddle 3.3.1, paddle2onnx 2.1.0, onnxslim (opset 12).
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import yaml
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
WORK = HERE / "_export_work"
WORK.mkdir(exist_ok=True)

PY = sys.executable  # run paddle2onnx / onnxslim from THIS interpreter's venv
BIN = Path(PY).parent  # venv bin/ holds the paddle2onnx + onnxslim console scripts
PADDLE2ONNX = str(BIN / "paddle2onnx")
ONNXSLIM = str(BIN / "onnxslim")

# (HF repo, fixed static input shape, output onnx name in the app folder)
DET_REPO = "PaddlePaddle/PP-OCRv5_mobile_det"
DET_SHAPE = (1, 3, 736, 736)
DET_ONNX = "ppocrv5_mobile_det.onnx"

REC_REPO = "PaddlePaddle/latin_PP-OCRv5_mobile_rec"
REC_SHAPE = (1, 3, 48, 320)
REC_ONNX = "latin_ppocrv5_mobile_rec.onnx"

OPSET = 12


def dims(t):
    return [(d.dim_value if d.HasField("dim_value") else d.dim_param)
            for d in t.type.tensor_type.shape.dim]


def assert_static(onnx_path: str, tag: str):
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    bad = []
    for vi in list(m.graph.input) + list(m.graph.output):
        for d in vi.type.tensor_type.shape.dim:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                bad.append((vi.name, dims(vi)))
    if bad:
        raise SystemExit(f"[FAIL] {tag}: dynamic/undefined axes remain: {bad}")
    print(f"[static-ok] {tag}: no dynamic axes")
    return m


def export_one(repo: str, shape, onnx_name: str, tag: str) -> str:
    local = snapshot_download(repo, local_dir=str(WORK / repo.split("/")[-1]))
    print(f"[download] {repo} -> {local}")

    raw = WORK / f"{tag}_raw.onnx"
    subprocess.run(
        [PADDLE2ONNX,
         "--model_dir", local,
         "--model_filename", "inference.json",
         "--params_filename", "inference.pdiparams",
         "--save_file", str(raw),
         "--opset_version", str(OPSET)],
        check=True,
    )
    print(f"[paddle2onnx] {tag} -> {raw}")

    dst = HERE / onnx_name
    shape_str = ",".join(str(s) for s in shape)
    subprocess.run(
        [ONNXSLIM, str(raw), str(dst),
         "--input-shapes", f"x:{shape_str}"],
        check=True,
    )
    print(f"[onnxslim static] {tag} -> {dst}  (input x:{shape_str})")

    m = assert_static(str(dst), tag)
    print(f"== {tag} ACTUAL ONNX I/O ==")
    for i in m.graph.input:
        print(f"  input  {i.name}: {dims(i)}")
    for o in m.graph.output:
        print(f"  output {o.name}: {dims(o)}")

    sample = np.random.rand(*shape).astype(np.float32)
    sample_name = onnx_name.replace(".onnx", "_sample_input.npy")
    np.save(HERE / sample_name, sample)
    print(f"[sample] {sample_name} shape={sample.shape} dtype={sample.dtype}\n")
    return local


def dump_charset(rec_local: str):
    """Extract the recognizer CTC character dictionary -> latin_charset.txt (Dart asset).

    PaddleOCR CTCLabelDecode builds its label list as:
        index 0            -> 'blank'  (CTC blank; skipped when decoding)
        index 1 .. N       -> character_dict[0 .. N-1]   (N = 836 here)
        index N+1          -> ' '  (space, appended by PaddleOCR)
    So the recognizer output class count is N + 2 = 838. latin_charset.txt below holds the
    RAW N=836-entry dictionary (one char per line, order preserved); Dart must prepend the
    blank at index 0 and append a space at the final index to reconstruct the full mapping.
    """
    with open(Path(rec_local) / "inference.yml") as f:
        y = yaml.safe_load(f)
    cd = y["PostProcess"]["character_dict"]
    out = HERE / "latin_charset.txt"
    with open(out, "w") as f:
        for c in cd:
            f.write(c + "\n")
    print(f"[charset] latin_charset.txt: {len(cd)} chars "
          f"(full CTC classes = {len(cd) + 2}: blank@0 + {len(cd)} + space@{len(cd) + 1})")


def main():
    export_one(DET_REPO, DET_SHAPE, DET_ONNX, "detector")
    rec_local = export_one(REC_REPO, REC_SHAPE, REC_ONNX, "recognizer")
    dump_charset(rec_local)


if __name__ == "__main__":
    main()
