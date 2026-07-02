"""
export.py — Re-runnable ONNX export for RetinaDRGrade.

Model: Kontawat/vit-diabetic-retinopathy-classification
  ViT-base (ViTForImageClassification), 5-class diabetic-retinopathy SEVERITY grade.
  Output = raw logits[1,5] over grades {0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative}.
  id2label is the IDENTITY map {0:'0',1:'1',2:'2',3:'3',4:'4'}, so argmax index == canonical grade.
  License: apache-2.0.

Preprocessing the model expects (ViTImageProcessor / preprocessor_config.json):
  - resize to 224x224, bilinear (PIL resample=2)
  - RGB channel order
  - rescale /255  (rescale_factor = 1/255)
  - normalize with mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]   -> pixels mapped to [-1, 1]
  - layout NCHW, float32, shape [1,3,224,224]

Output: float32 logits[1,5] (RAW — apply softmax + argmax downstream). Nothing baked in.

Export notes (hard-won, keep these):
  - transformers' DEFAULT attention is SDPA, whose ONNX op requires opset >= 14.
    To get a clean opset-12 export we force `attn_implementation="eager"`, which is
    MATHEMATICALLY IDENTICAL (same softmax attention, just the explicit matmul path).
  - `dynamo=False` uses the legacy TorchScript exporter, which reliably supports opset 12.
  - Static shapes only: dynamic_axes=None. Melange wants fixed dims.
  - half=False: keep the ONNX fp32; Melange decides precision server-side.
  - Artifact is ~343 MB fp32 (ViT-base). Flag as a first-launch download / on-device
    size consideration for Melange-fit.

Run:
  pip install torch transformers onnx onnxruntime numpy
  python export.py
"""
import numpy as np
import torch
import onnx
import onnxruntime as ort
from transformers import ViTForImageClassification

# Load straight from the Hugging Face Hub (re-runnable anywhere). To export from a
# local checkpoint instead, point MODEL at a directory containing config.json +
# pytorch_model.bin (e.g. the validated copy in
# ../RetinaDRScreen/_eval/vit-kontawat/model).
MODEL = "Kontawat/vit-diabetic-retinopathy-classification"
OUT = "vit-base-dr-grade.onnx"

# attn_implementation="eager" -> opset-12-compatible (SDPA needs opset >= 14).
model = ViTForImageClassification.from_pretrained(MODEL, attn_implementation="eager")
model.eval()


class Wrap(torch.nn.Module):
    """Return only logits[1,5] so the ONNX has a single clean output."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).logits


w = Wrap(model)
dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)

torch.onnx.export(
    w,
    (dummy,),
    OUT,
    input_names=["pixel_values"],
    output_names=["logits"],
    opset_version=12,
    do_constant_folding=True,
    dynamic_axes=None,  # static shape, no dynamic axes
    dynamo=False,       # legacy TorchScript exporter (reliable opset-12 support)
)
print("exported", OUT)

# ---- Verify the artifact ----
m = onnx.load(OUT)
onnx.checker.check_model(m)
print("onnx.checker: PASS")
print("ir_version:", m.ir_version, "opset:", m.opset_import[0].version)
for i in m.graph.input:
    print("input ", i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output:
    print("output", o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])

# ---- Parity: torch vs onnxruntime on identical random input ----
with torch.no_grad():
    t = w(dummy).numpy()
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
o = sess.run(["logits"], {"pixel_values": dummy.numpy()})[0]
print("max abs diff torch vs ort:", float(np.max(np.abs(t - o))))
print("torch argmax:", int(t.argmax()), "ort argmax:", int(o.argmax()))
