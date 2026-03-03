#!/usr/bin/env python3
"""
Export script for meta-llama/Llama-Prompt-Guard-2-86M.
Exports to TorchScript, ExportedProgram, and ONNX (FP32). Saves sample inputs as ordered .npy files.
"""
import os
import sys

# Add project root for mentat imports when running from script/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(MODEL_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

def _get_hf_token():
    """Use HF_TOKEN env var or huggingface_hub stored token (huggingface-cli login). No fallback."""
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    try:
        from huggingface_hub import get_token
        t = get_token()
        if t:
            return t
    except Exception:
        pass
    return None

HF_TOKEN = _get_hf_token()
MODEL_ID = os.environ.get("OVERRIDE_MODEL_ID", "meta-llama/Llama-Prompt-Guard-2-86M")
BATCH_SIZE = 1
SEQ_LENGTH = 128  # static shape; must match ZeticTensorFactory.seqLen and app modelMaxTokens
MODEL_EXPORT_DIR = os.path.join(MODEL_DIR, "model_export")
MODEL_INPUTS_DIR = os.path.join(MODEL_DIR, "model_inputs")
ONNX_OPSET = 17
MAX_ONNX_BYTES = 2 * 1024 * 1024 * 1024  # 2GB for Protobuf


def get_model_and_config():
    from transformers import AutoConfig, AutoModelForSequenceClassification

    print(f"[Export] Loading config for {MODEL_ID}...")
    token_kw = {"token": HF_TOKEN} if HF_TOKEN else {}
    config = AutoConfig.from_pretrained(MODEL_ID, **token_kw)
    print(f"[Export] Loading model (FP32)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        **token_kw,
    )
    model.eval()
    return model, config


def prepare_and_save_inputs(config):
    os.makedirs(MODEL_INPUTS_DIR, exist_ok=True)
    vocab_size = getattr(config, "vocab_size", 128000)
    # int32 for NPU/backend compatibility
    input_ids = torch.randint(0, min(vocab_size, 32000), (BATCH_SIZE, SEQ_LENGTH), dtype=torch.int32)
    attention_mask = torch.ones(BATCH_SIZE, SEQ_LENGTH, dtype=torch.int32)
    inputs = [input_ids, attention_mask]
    names = ["input_ids", "attention_mask"]
    for i, (name, t) in enumerate(zip(names, inputs)):
        path = os.path.join(MODEL_INPUTS_DIR, f"{i}_{name}.npy")
        arr = t.numpy()
        if arr.dtype != np.int32:
            arr = arr.astype(np.int32)
        np.save(path, arr)
        print(f"[Export] Saved {path} (dtype={arr.dtype})")
    return inputs, names


class _TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


def export_torchscript(model, inputs, input_names):
    from mentat.ztc.convert.coreml_deberta_patches import deberta_coreml_trace_patches

    path = os.path.join(MODEL_EXPORT_DIR, "model.pt")
    os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
    wrapper = _TraceWrapper(model)
    wrapper.eval()
    with deberta_coreml_trace_patches():
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, tuple(inputs), check_trace=False, strict=False)
            ref_logits = wrapper(*tuple(inputs))
            trace_logits = traced(*tuple(inputs))
            if not np.allclose(ref_logits.numpy(), trace_logits.numpy(), atol=1e-5, rtol=1e-4):
                print("[Export] WARNING: TorchScript output differs from eager on export inputs.")
        traced.save(path)
    print(f"[Export] TorchScript saved to {path}")
    return path


def export_exported_program(model, inputs, input_names):
    from mentat.ztc.convert.coreml_deberta_patches import deberta_coreml_trace_patches

    path = os.path.join(MODEL_EXPORT_DIR, "model.pt2")
    os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
    try:
        from torch.export import export, save

        wrapper = _TraceWrapper(model)
        wrapper.eval()
        inps = [
            torch.from_numpy(np.load(os.path.join(MODEL_INPUTS_DIR, f"{i}_{input_names[i]}.npy")))
            for i in range(len(input_names))
        ]
        with deberta_coreml_trace_patches():
            ep = export(wrapper, tuple(inps), strict=False)
            save(ep, path)
        print(f"[Export] ExportedProgram saved to {path}")
        return path
    except Exception as e:
        print(f"[Export] ExportedProgram export failed: {e}")
        return None


def export_onnx(model, inputs, input_names):
    path = os.path.join(MODEL_EXPORT_DIR, "model.onnx")
    os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
    wrapper = _TraceWrapper(model)
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(inputs),
            path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
        )
    print(f"[Export] ONNX saved to {path}")
    return path


def shrink_onnx_embedding_if_needed(onnx_path):
    if not os.path.exists(onnx_path):
        return
    size_bytes = os.path.getsize(onnx_path)
    if size_bytes <= MAX_ONNX_BYTES:
        print(f"[Export] ONNX size {size_bytes / (1024**2):.2f} MB <= 2GB, no surgery.")
        return
    print(f"[Export] ONNX size {size_bytes / (1024**2):.2f} MB > 2GB, applying embedding FP16 surgery...")
    try:
        import onnx
        from onnx import TensorProto
        import onnx_graphsurgeon as gs
    except ImportError as e:
        print(f"[Export] Cannot run ONNX surgery (missing deps): {e}")
        return
    model_proto = onnx.load(onnx_path)
    graph = gs.import_onnx(model_proto)
    gather_node = None
    largest_const = None
    largest_numel = 0
    for node in graph.nodes:
        if node.op != "Gather" or len(node.inputs) < 1:
            continue
        data_in = node.inputs[0]
        if not isinstance(data_in, gs.Constant) or data_in.values.dtype != np.float32:
            continue
        numel = data_in.values.size
        if numel > largest_numel:
            largest_numel = numel
            largest_const = data_in
            gather_node = node
    if gather_node is None or largest_const is None:
        print("[Export] No suitable Gather/embedding initializer found for surgery.")
        return
    largest_const.values = largest_const.values.astype(np.float16)
    gather_out = gather_node.outputs[0]
    cast_out_name = f"{gather_out.name}_cast_fp32"
    cast_out = gs.Variable(name=cast_out_name, dtype=np.float32, shape=gather_out.shape)
    cast_node = gs.Node(op="Cast", name=f"Cast_{gather_out.name}", attrs={"to": int(TensorProto.FLOAT)}, inputs=[gather_out], outputs=[cast_out])
    graph.nodes.append(cast_node)
    for node in graph.nodes:
        for j, inp in enumerate(node.inputs):
            if inp == gather_out:
                node.inputs[j] = cast_out
    if gather_out in graph.outputs:
        idx = graph.outputs.index(gather_out)
        graph.outputs[idx] = cast_out
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), onnx_path)
    print(f"[Export] ONNX embedding surgery done. Saved to {onnx_path}")


def main():
    print("[Export] Starting export pipeline for", MODEL_ID)
    model, config = get_model_and_config()
    inputs, input_names = prepare_and_save_inputs(config)
    os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)

    export_torchscript(model, inputs, input_names)
    export_exported_program(model, inputs, input_names)
    onnx_path = export_onnx(model, inputs, input_names)
    if onnx_path and os.path.getsize(onnx_path) > MAX_ONNX_BYTES:
        shrink_onnx_embedding_if_needed(onnx_path)

    print("[Export] Done.")


if __name__ == "__main__":
    main()
