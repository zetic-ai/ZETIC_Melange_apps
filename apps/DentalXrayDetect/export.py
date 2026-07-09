"""
DentalXrayDetect — Stage 0 export recipe (re-runnable).

Model:   liodon-ai/dental-panoramic-detector  (best.pt)
Weights: standard Ultralytics YOLO11n, 3 classes {caries, periapical_lesion, impacted_tooth},
         trained @ imgsz 640 on 9,928 dental panoramic radiographs
         (DENTEX + OralXrays-9). mAP50 0.622 / mAP50-95 0.406 on DENTEX val.
LICENSE: cc-by-nc-4.0  ***NON-COMMERCIAL***  — see model_selection.md license flag.

Family recipe: Ultralytics YOLO -> ONNX, STATIC shapes (dynamic=False), opset 12,
         half=False (Melange handles precision). Identical recipe to PyroGuard /
         AerialDetectYOLO; only the checkpoint + imgsz differ (imgsz=640 here).

Note: the repo already ships a `best.onnx`, but we re-export from `best.pt` ourselves to
      GUARANTEE opset 12 + static shapes (the shipped ONNX's opset/axes are unverified).

Deps: pip install ultralytics onnx huggingface_hub numpy
      (This machine: used a Python 3.9 venv — torch 2.5.1 / ultralytics 8.4.84 install
      cleanly there. Homebrew python@3.14 lacks torch wheels; if you must use it, note the
      pyexpat/expat loader quirk documented in AerialDetectYOLO/export.py.)
"""
import os
import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

REPO_ID  = "liodon-ai/dental-panoramic-detector"   # cc-by-nc-4.0 (NON-COMMERCIAL)
WEIGHT   = "best.pt"                                # standard Ultralytics YOLO11n, 3 classes
IMGSZ    = 640                                      # training resolution; divisible by 32; STATIC
OPSET    = 12
ONNX_OUT = os.path.join(HERE, "dentalxray-yolo11n.onnx")

def main():
    # 1. Pull the chosen dental-pathology checkpoint from Hugging Face.
    pt_path = hf_hub_download(REPO_ID, WEIGHT)
    model = YOLO(pt_path)
    print("Loaded:", REPO_ID, WEIGHT)
    print("Task:", model.task, "| Classes:", model.names)

    # 2. Export to ONNX with the static-shape YOLO recipe.
    exported = model.export(
        format="onnx",
        imgsz=IMGSZ,        # 640x640 static (training resolution)
        opset=OPSET,
        simplify=True,
        dynamic=False,      # STATIC SHAPES OR BUST
        half=False,         # Melange decides precision; keep ONNX fp32
    )
    if os.path.abspath(exported) != os.path.abspath(ONNX_OUT):
        os.replace(exported, ONNX_OUT)
    print("Exported ONNX:", ONNX_OUT)

    # 3. Inspect ACTUAL ONNX I/O shapes (do not assume).
    m = onnx.load(ONNX_OUT)
    def shp(t):
        return [d.dim_value if (d.dim_value > 0) else (d.dim_param or "?")
                for d in t.type.tensor_type.shape.dim]
    in_t, out_t = m.graph.input[0], m.graph.output[0]
    in_shape, out_shape = shp(in_t), shp(out_t)
    print("ONNX input :", in_t.name, in_shape)
    print("ONNX output:", out_t.name, out_shape)
    assert all(isinstance(d, int) for d in in_shape),  f"dynamic input dim: {in_shape}"
    assert all(isinstance(d, int) for d in out_shape), f"dynamic output dim: {out_shape}"
    print("STATIC SHAPES CONFIRMED (no dynamic axes).")

    # 4. Generate sample_input.npy — random noise of the right shape/dtype.
    #    (Melange only needs shape+dtype to compile; this is NOT a validation of outputs.)
    sample = np.random.rand(*in_shape).astype(np.float32)
    np.save(os.path.join(HERE, "sample_input.npy"), sample)
    print("Wrote sample_input.npy:", sample.shape, sample.dtype)

if __name__ == "__main__":
    main()
