#!/usr/bin/env python3
"""
Stage-0 export recipe — VehiclePlateYOLO (ZETIC Melange demo).

Family: YOLO (Ultralytics) — reuses the PyroGuard YOLO recipe verbatim.
Winner model: Koushim/yolov8-license-plate-detection
  weight file: best.pt  (YOLOv8 single-class license-plate detector)
  license: MIT  (GTM-clean; selected at GATE 0 over the AGPL technical pick
                 morsetechlab/yolov11-license-plate-detection — see model_selection.md)

What this script does, re-runnably:
  1. Download the winner .pt weights from the Hugging Face Hub.
  2. Export to ONNX with STATIC shapes (dynamic=False), opset 12, simplified,
     no half precision (Melange owns precision).
  3. Copy <model>.onnx into this app folder.
  4. Emit sample_input.npy = np.random.rand(1,3,640,640).float32 (shape/dtype only).
  5. Print the ACTUAL input/output tensor shapes read back from the exported ONNX.

Setup (no ultralytics in repo):
  uv venv --python 3.12 venv && source venv/bin/activate
  uv pip install ultralytics onnx onnxslim huggingface_hub numpy
  python export.py
"""
import shutil
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent

HF_REPO = "Koushim/yolov8-license-plate-detection"
HF_FILE = "best.pt"   # YOLOv8 single-class license-plate detector (MIT)
IMGSZ = 640
OPSET = 12
ONNX_NAME = "koushim-yolov8-license-plate.onnx"


def main() -> None:
    # 1. Fetch winner weights from HF
    pt_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    print(f"[download] {HF_REPO}/{HF_FILE} -> {pt_path}")

    # 2. Export — STATIC shapes, opset 12, simplified, FP32 (PyroGuard recipe)
    model = YOLO(pt_path)
    print(f"[model] names = {model.names}")
    exported = model.export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        simplify=True,
        dynamic=False,   # static shapes or bust
        half=False,      # Melange handles precision
    )
    print(f"[export] {exported}")

    # 3. Place ONNX in the app folder
    dst = HERE / ONNX_NAME
    shutil.copyfile(exported, dst)
    print(f"[onnx] -> {dst}")

    # 4. sample_input.npy — random noise, correct shape/dtype only
    sample = np.random.rand(1, 3, IMGSZ, IMGSZ).astype(np.float32)
    np.save(HERE / "sample_input.npy", sample)
    print(f"[sample] sample_input.npy shape={sample.shape} dtype={sample.dtype}")

    # 5. Read back ACTUAL tensor shapes from the exported ONNX
    m = onnx.load(str(dst))

    def dims(t):
        return [d.dim_value if (d.dim_value or not d.dim_param) else d.dim_param
                for d in t.type.tensor_type.shape.dim]

    print("\n== ACTUAL ONNX I/O ==")
    for i in m.graph.input:
        print(f"  input  {i.name}: {dims(i)}")
    for o in m.graph.output:
        print(f"  output {o.name}: {dims(o)}")


if __name__ == "__main__":
    main()
