"""
export.py — SensorForecastTS (time-series forecasting family recipe, first of its kind)

Exports amazon/chronos-bolt-tiny to a STATIC-shape ONNX for ZETIC Melange.

FAMILY RECIPE NOTES (reuse for any Chronos-Bolt size):
  Chronos-Bolt is a T5-style encoder-decoder, but unlike original Chronos it is
  NOT autoregressive: one forward pass emits the full multi-step quantile
  forecast. That makes it exportable as a single static graph. Three things
  fight a clean static export and are handled below:
    1. `encode()` has a dynamic context-length check (`if context.shape[-1] >
       context_length: ...`) -> monkeypatched out; we guarantee the static
       context length at the app layer instead.
    2. `torch.full(...)` for the REG/decoder-start token ids traces with an
       implicit dtype that breaks embedding lookup in ONNX -> patched to
       dtype=torch.long explicitly.
    3. The `Patch` module pads when length % patch_size != 0 (control flow) ->
       replaced with StaticPatch (pure unfold); CONTEXT_LENGTH must be a
       multiple of input_patch_size (16), which 512 is.
    4. `InstanceNorm` uses `aten::nanmean` (NaN-tolerant scaling), which the
       ONNX exporter does not support at ANY opset -> patched to plain
       mean/std, and the observed-mask is hardcoded to ones. CONSEQUENCE: the
       exported graph does NOT support NaN left-padding; the app MUST feed a
       full window of 512 real values (this is also NPU-safer: no isnan ops).
  This mirrors the recipe ZETIC themselves used for their Chronos-Bolt demo
  (apps/ChronosTimeSeries/prepare/extract_chronos.py, Team_ZETIC/Chronos-balt-tiny),
  which is proven to convert on Melange.

  Opset ladder (measured, torch 2.12 legacy tracer): before patches 3+4, ALL
  of 12/13/14 fail (aten::nanmean unsupported; unfold mis-decomposed). After
  the patches, 12, 13 and 14 ALL export cleanly with identical outputs
  (max |onnx-torch| = 1.5e-05). We ship opset 12 — the family's known-good
  Melange baseline (same as the YOLO recipe).

INPUT  : context        float32[1, 512]   raw sensor values, unnormalized
                                          (model does instance norm in-graph).
                                          MUST be a full window of real values;
                                          NaN padding is NOT supported by this
                                          export (see note 4 above).
OUTPUT : quantile_preds float32[1, 9, 64] 9 quantiles (0.1 ... 0.9) x 64 future
                                          steps, in the ORIGINAL data scale
                                          (de-normalized in-graph). Median is
                                          index 4 on dim 1.

Run:  python export.py
Deps: chronos-forecasting, torch, onnx, onnxruntime, numpy
"""

import os

import numpy as np
import torch
import torch.nn as nn

from chronos import ChronosBoltPipeline
from chronos.chronos_bolt import ChronosBoltModelForForecasting

HF_REPO = "amazon/chronos-bolt-tiny"
CONTEXT_LENGTH = 512          # static; multiple of input_patch_size=16
PREDICTION_LENGTH = 64        # fixed by the checkpoint
NUM_QUANTILES = 9             # fixed by the checkpoint: 0.1 ... 0.9
OPSET = 12                    # known-good baseline; 13/14 also work (see header)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(OUT_DIR, "chronos-bolt-tiny-ctx512.onnx")
SAMPLE_PATH = os.path.join(OUT_DIR, "sample_input.npy")


# --------------------------------------------------------------------------
# Monkeypatches: remove dynamic control flow so tracing yields a static graph
# --------------------------------------------------------------------------
def patched_instance_norm_forward(self, x: torch.Tensor, loc_scale=None):
    """NaN-free InstanceNorm: aten::nanmean is unexportable to ONNX at any
    opset (verified: opsets 12/13/14 all fail on it). The app guarantees a
    full real-valued window, so plain mean/std is mathematically identical."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if loc_scale is None:
        loc = x.mean(dim=-1, keepdim=True)
        scale = (x - loc).square().mean(dim=-1, keepdim=True).sqrt()
        scale = torch.where(scale == 0, self.eps, scale)
    else:
        loc, scale = loc_scale
    scaled_x = (x - loc) / scale
    if self.use_arcsinh:
        scaled_x = torch.arcsinh(scaled_x)
    return scaled_x.to(orig_dtype), (loc, scale)


def patched_encode(self, context: torch.Tensor, mask: torch.Tensor = None):
    # Hardcoded all-observed mask: the app feeds a full window of real values
    # (no NaN padding). Removes isnan/logical_not from the graph.
    mask = torch.ones_like(context)
    # REMOVED: dynamic "context longer than context_length" truncation branch.
    # The app guarantees exactly CONTEXT_LENGTH samples.
    batch_size, _ = context.shape

    context, loc_scale = self.instance_norm(context)
    context = context.to(self.dtype)
    mask = mask.to(self.dtype)

    patched_context = self.patch(context)
    patched_mask = torch.nan_to_num(self.patch(mask), nan=0.0)
    patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
    patched_context = torch.cat([patched_context, patched_mask], dim=-1)

    attention_mask = patched_mask.sum(dim=-1) > 0

    input_embeds = self.input_patch_embedding(patched_context)

    if self.chronos_config.use_reg_token:
        reg_input_ids = torch.full(
            (batch_size, 1),
            self.config.reg_token_id,
            device=input_embeds.device,
            dtype=torch.long,  # explicit long dtype (trace-time fix)
        )
        reg_embeds = self.shared(reg_input_ids)
        input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
        attention_mask = torch.cat(
            [attention_mask.to(self.dtype), torch.ones_like(reg_input_ids).to(self.dtype)],
            dim=-1,
        )

    encoder_outputs = self.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
    return encoder_outputs[0], loc_scale, input_embeds, attention_mask


def patched_decode(self, input_embeds, attention_mask, hidden_states, output_attentions=False):
    batch_size = input_embeds.shape[0]
    decoder_input_ids = torch.full(
        (batch_size, 1),
        self.config.decoder_start_token_id,
        device=input_embeds.device,
        dtype=torch.long,  # explicit long dtype (trace-time fix)
    )
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        output_attentions=output_attentions,
        return_dict=True,
    )
    return decoder_outputs.last_hidden_state


class StaticPatch(nn.Module):
    """Patch module without the length-remainder padding branch.

    NOTE: implemented as a pure reshape, NOT x.unfold(). The legacy ONNX
    exporter mis-decomposes unfold (emits [1,16,32] slices without the final
    transpose instead of [1,32,16] -> downstream MatMul shape error, verified
    on torch 2.12 at opsets 12/13/14). With patch_stride == patch_size
    (non-overlapping patches, true for all Chronos-Bolt sizes) unfold IS a
    reshape, which exports as a single static Reshape op.
    """

    def __init__(self, patch_size: int, patch_stride: int):
        super().__init__()
        assert patch_size == patch_stride, "reshape trick needs non-overlapping patches"
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, length = x.shape
        return x.reshape(b, length // self.patch_size, self.patch_size)


class ChronosExportWrapper(nn.Module):
    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model

    def forward(self, context):
        # [batch, num_quantiles, prediction_length], original data scale
        return self.inner_model(context=context).quantile_preds


def main():
    print(f"[1/6] Loading {HF_REPO} ...")
    from chronos.chronos_bolt import InstanceNorm

    InstanceNorm.forward = patched_instance_norm_forward
    ChronosBoltModelForForecasting.encode = patched_encode
    ChronosBoltModelForForecasting.decode = patched_decode

    pipeline = ChronosBoltPipeline.from_pretrained(
        HF_REPO, device_map="cpu", torch_dtype=torch.float32
    )
    model = pipeline.model
    model.eval()
    model.config.use_cache = False
    if hasattr(model, "decoder"):
        model.decoder.config.use_cache = False

    assert CONTEXT_LENGTH % model.chronos_config.input_patch_size == 0
    assert model.chronos_config.prediction_length == PREDICTION_LENGTH
    assert len(model.chronos_config.quantiles) == NUM_QUANTILES
    model.patch = StaticPatch(
        patch_size=model.chronos_config.input_patch_size,
        patch_stride=model.chronos_config.input_patch_stride,
    )

    wrapper = ChronosExportWrapper(model)
    wrapper.eval()

    print("[2/6] Reference forward pass (torch) ...")
    # A realistic-ish trace input: noisy sine, full window, no NaNs.
    t = np.arange(CONTEXT_LENGTH, dtype=np.float32)
    trace_ctx = torch.tensor(
        (50 + 10 * np.sin(2 * np.pi * t / 64) + np.random.default_rng(0).normal(0, 1, CONTEXT_LENGTH)).astype(
            np.float32
        )
    ).unsqueeze(0)
    with torch.no_grad():
        ref_out = wrapper(trace_ctx)
    assert ref_out.shape == (1, NUM_QUANTILES, PREDICTION_LENGTH), ref_out.shape

    print(f"[3/6] Exporting ONNX (opset {OPSET}, static shapes, no dynamic_axes) ...")
    torch.onnx.export(
        wrapper,
        (trace_ctx,),
        ONNX_PATH,
        input_names=["context"],
        output_names=["quantile_preds"],
        opset_version=OPSET,
        dynamic_axes=None,  # fully static: batch=1, context=512
        do_constant_folding=True,
        dynamo=False,  # legacy tracer: static graph, no onnxscript rewrites
    )

    print("[4/6] Checking ONNX: checker + NO dynamic dims allowed ...")
    import onnx

    m = onnx.load(ONNX_PATH)
    onnx.checker.check_model(m)
    m = onnx.shape_inference.infer_shapes(m)

    def dims_of(vi):
        return [
            (d.dim_value if d.HasField("dim_value") else d.dim_param)
            for d in vi.type.tensor_type.shape.dim
        ]

    for vi in list(m.graph.input) + list(m.graph.output):
        dims = dims_of(vi)
        assert all(isinstance(d, int) and d > 0 for d in dims), f"dynamic dim in {vi.name}: {dims}"
        print(f"    {vi.name}: {dims}")
    onnx.save(m, ONNX_PATH)

    print("[5/6] Verifying ONNX vs torch with onnxruntime ...")
    import onnxruntime as ort

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"context": trace_ctx.numpy()})[0]
    diff = float(np.abs(onnx_out - ref_out.numpy()).max())
    print(f"    max |onnx - torch| = {diff:.3e}")
    assert diff < 1e-3, "ONNX output diverges from torch reference"

    print("[6/6] Writing sample_input.npy (random noise, correct shape/dtype) ...")
    sample = np.random.rand(1, CONTEXT_LENGTH).astype(np.float32)
    np.save(SAMPLE_PATH, sample)

    print("\nDone.")
    print(f"  model : {ONNX_PATH} ({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)")
    print(f"  sample: {SAMPLE_PATH} shape={sample.shape} dtype={sample.dtype}")


if __name__ == "__main__":
    main()
