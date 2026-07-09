"""
ShelfScanYOLO — Stage 0 export recipe (re-runnable).

Model:   chistopat/sku110k-yolo11-object-detector
Weights: weights/sku110k-yolo11-s640.pt  (standard Ultralytics YOLO11s, trained at imgsz 640
         on SKU-110K — the canonical DENSE retail-shelf dataset; 1 class "object" = one
         product facing / SKU on a packed shelf. This is exactly the shelf-execution task
         Infilect/Trax/Shopic solve. mAP50 0.927, mAP50-95 0.577 on the SKU-110K test split.)
Family recipe: Ultralytics YOLO -> ONNX, STATIC shapes (dynamic=False), opset 12,
         simplify=True, half=False (Melange handles precision). Same recipe as PyroGuard.

Why re-export when the repo already ships an .onnx: we pin opset 12 (the known-good Melange
opset from PyroGuard) and confirm the fixed-shape graph ourselves, rather than trusting an
upstream export of unknown opset.

Environment note (this machine): Homebrew python@3.14 ships a pyexpat built against a newer
expat than /usr/lib/libexpat.1.dylib, so pip/torch import fails until you point the loader at
Homebrew's expat. Run with:
    DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/expat/2.8.2/lib python export.py
(adjust the expat version path if different; `brew install expat` provides it).

Deps: pip install ultralytics onnx onnxslim huggingface_hub numpy onnxruntime dill
"""
import os
import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

REPO_ID  = "chistopat/sku110k-yolo11-object-detector"     # license: other (SKU-110K terms)
WEIGHT   = "weights/sku110k-yolo11-s640.pt"               # standard YOLO11s, trained @ imgsz 640
IMGSZ    = 640                                            # divisible by 32; STATIC, fixed
OPSET    = 12
ONNX_OUT = os.path.join(HERE, "shelfscan-yolo11s-sku110k.onnx")


def main():
    # 1. Pull the chosen SKU-110K checkpoint from Hugging Face.
    pt_path = hf_hub_download(REPO_ID, WEIGHT)
    model = YOLO(pt_path)
    print("Loaded:", REPO_ID, WEIGHT)
    print("Classes:", model.names)

    # 2. Export to ONNX with the static-shape YOLO recipe (opset 12).
    exported = model.export(
        format="onnx",
        imgsz=IMGSZ,        # 640x640 static
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
    in_t  = m.graph.input[0]
    out_t = m.graph.output[0]
    in_shape  = shp(in_t)
    out_shape = shp(out_t)
    print("ONNX input :", in_t.name, in_shape)
    print("ONNX output:", out_t.name, out_shape)
    assert all(isinstance(d, int) for d in in_shape),  f"dynamic input dim: {in_shape}"
    assert all(isinstance(d, int) for d in out_shape), f"dynamic output dim: {out_shape}"
    print("STATIC SHAPES CONFIRMED (no dynamic axes).")

    # 4. Generate sample_input.npy — random noise of the right shape/dtype.
    #    (Melange only needs shape+dtype to compile; this is NOT a validation of outputs.)
    sample = np.random.rand(*in_shape).astype(np.float32)
    npy_out = os.path.join(HERE, "sample_input.npy")
    np.save(npy_out, sample)
    print("Wrote sample_input.npy:", sample.shape, sample.dtype)


if __name__ == "__main__":
    main()
