#!/usr/bin/env python3
"""
Trace Qwen3-TTS-12Hz-0.6B-CustomVoice.
Based on working V2 prototype.
"""

import os
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn

# --- GLOBAL PATCHES (MUST BE BEFORE IMPORTS) ---
import math
import torch.nn.functional as F
import transformers.utils

if not hasattr(transformers.utils, "auto_docstring"):

    def dummy_auto_docstring(*args, **kwargs):
        # Case 1: @auto_docstring class Foo
        if len(args) == 1 and isinstance(args[0], type):
            return args[0]
        # Case 2: @auto_docstring(...) class Foo
        return lambda cls: cls

    transformers.utils.auto_docstring = dummy_auto_docstring

if not hasattr(transformers.utils, "can_return_tuple"):

    def dummy_can_return(*args, **kwargs):
        if len(args) > 0 and callable(args[0]):
            return args[0]
        return True

    transformers.utils.can_return_tuple = dummy_can_return

import transformers.integrations

if not hasattr(transformers.integrations, "use_kernel_forward_from_hub"):

    def dummy_use_kernel(*args, **kwargs):
        return lambda cls: cls

    transformers.integrations.use_kernel_forward_from_hub = dummy_use_kernel

    transformers.integrations.use_kernel_forward_from_hub = dummy_use_kernel

import sys
from types import ModuleType

# Patch masking_utils for broken transformers dev version
# Patch masking_utils for broken transformers dev version
import sys
from types import ModuleType
import transformers

try:
    import transformers.masking_utils
except ImportError:
    m = ModuleType("transformers.masking_utils")

    def dummy_create_masks(*args, **kwargs):
        return None

    def dummy_combine(*args, **kwargs):
        return args[0]

    def dummy_causal(*args, **kwargs):
        return None

    m.create_masks_for_generate = dummy_create_masks
    m.combine_mask_and_class_mask = dummy_combine
    m.create_causal_mask = dummy_causal
    m.create_sliding_window_causal_mask = dummy_causal
    sys.modules["transformers.masking_utils"] = m
    transformers.masking_utils = m

import transformers.masking_utils

if not hasattr(transformers.masking_utils, "create_masks_for_generate"):

    def dummy_create_masks(*args, **kwargs):
        return None

    transformers.masking_utils.create_masks_for_generate = dummy_create_masks

if "transformers.modeling_layers" not in sys.modules:
    m = ModuleType("transformers.modeling_layers")

    class GradientCheckpointingLayer(torch.nn.Module):
        pass

    class GradientCheckpointingLayer(torch.nn.Module):
        pass

    m.GradientCheckpointingLayer = GradientCheckpointingLayer
    sys.modules["transformers.modeling_layers"] = m

import transformers.modeling_rope_utils

if not hasattr(transformers.modeling_rope_utils, "dynamic_rope_update"):
    transformers.modeling_rope_utils.dynamic_rope_update = lambda f: f

import transformers.utils.generic

if not hasattr(transformers.utils.generic, "check_model_inputs"):

    def dummy_check_model_inputs(*args, **kwargs):
        return lambda f: f

    transformers.utils.generic.check_model_inputs = dummy_check_model_inputs

import transformers.configuration_utils

if not hasattr(transformers.configuration_utils, "layer_type_validation"):

    def dummy_layer_type_validation(*args, **kwargs):
        return lambda f: f

    transformers.configuration_utils.layer_type_validation = dummy_layer_type_validation

# 2. Manual SDPA Patch (Global)
# Decomposed Attention for robust ONNX export and to bypass 'aten::sdpa' issues.
original_sdpa = F.scaled_dot_product_attention


def manual_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    **kwargs,
):
    # Query shape: [batch, heads, L, head_dim]
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # GQA Handling: Expand K, V if heads mismatch (e.g. 16 vs 8)
    n_rep = query.size(1) // key.size(1)
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    # 1. Matmul: [batch, heads, L, S]
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    # 2. Apply Masks
    if is_causal:
        # Causal mask (Avoid tril/triu for Opset 11)
        # Create indices [L, 1] and [1, S]
        idx_row = torch.arange(L, device=query.device).view(-1, 1)
        idx_col = torch.arange(S, device=query.device).view(1, -1)
        # Mask Upper Triangle (col > row)
        is_upper = idx_col > idx_row
        attn_weight.masked_fill_(is_upper, -1e9)

    if attn_mask is not None:
        # attn_mask shape typically [batch, 1, L, S]
        # Handle GQA mismatch for mask if it was already 16 heads (should match weight now)
        if (
            attn_mask.ndim == 4
            and attn_mask.shape[1] != attn_weight.shape[1]
            and attn_mask.shape[1] > 1
        ):
            # Attempt to repeat interleave to match query heads
            ratio = attn_weight.shape[1] // attn_mask.shape[1]
            if ratio > 1:
                attn_mask = attn_mask.repeat_interleave(ratio, dim=1)

        if attn_mask.dtype == torch.bool:
            attn_weight.masked_fill_(attn_mask.logical_not(), -1e9)
        else:
            # Robust patch: Replace -inf with -1e9 in the INPUT mask
            attn_mask = torch.where(
                attn_mask == float("-inf"),
                torch.tensor(-1e9, dtype=attn_mask.dtype, device=attn_mask.device),
                attn_mask,
            )
            # additive mask (broadcasts automatically)
            attn_weight = attn_weight + attn_mask

    # 3. Softmax & Projection
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


# Apply Patch Globally
F.scaled_dot_product_attention = manual_sdpa
print("[PATCH] Applied manual_sdpa globally to torch.nn.functional.")


# 3. New Ones Patch (for CoreML)
# CoreML doesn't support 'new_ones'. We replace it with torch.ones(..., device=self.device, dtype=self.dtype).
# This is safe because new_ones is just convenience for that.
def patched_new_ones(self, *args, **kwargs):
    size = args
    if "size" in kwargs:
        size = kwargs.pop("size")
    elif len(args) == 1 and isinstance(args[0], (list, tuple, torch.Size)):
        size = args[0]

    # Extract dtype and device from self if not provided
    dtype = kwargs.get("dtype", self.dtype)
    device = kwargs.get("device", self.device)

    # Remove dtype/device from kwargs if present to avoid double arg
    if "dtype" in kwargs:
        del kwargs["dtype"]
    if "device" in kwargs:
        del kwargs["device"]

    # Force size to be ints (CoreML requires int shapes)
    if isinstance(size, (list, tuple, torch.Size)):
        size = [int(s) for s in size]
        if len(size) == 0:
            # CoreML workaround: Scalar ones (empty size) fails during conversion.
            # Force size to [1] to create a rank-1 logical scalar.
            size = [1]

    return torch.ones(size, dtype=dtype, device=device, **kwargs)


torch.Tensor.new_ones = patched_new_ones
print("[PATCH] Applied patched_new_ones globally to torch.Tensor.")

# 4. Torch Ones Patch (Global)
# Force shape arguments to be integers.
original_torch_ones = torch.ones


def patched_torch_ones(*args, **kwargs):
    # Case 1: torch.ones(size, ...) where size is list/tuple/Size
    if len(args) > 0 and isinstance(args[0], (list, tuple, torch.Size)):
        shape = [int(s) for s in args[0]]
        return original_torch_ones(shape, *args[1:], **kwargs)

    # Case 2: torch.ones(*size, ...) where size is varargs of ints
    # We rely on inspecting args. If they are all numbers, cast them.
    # Note: torch.ones(2, 3) -> args=(2, 3)
    # torch.ones(2, 3, dtype=...) -> args=(2, 3), kwargs={dtype...}

    new_args = []
    for arg in args:
        if isinstance(arg, (int, float)):
            new_args.append(int(arg))
        elif isinstance(arg, (list, tuple, torch.Size)):
            # Should have been caught by Case 1 if it was the first arg,
            # but just in case it appears elsewhere or mixed
            new_args.append([int(x) for x in arg])
        else:
            new_args.append(arg)

    return original_torch_ones(*new_args, **kwargs)


torch.ones = patched_torch_ones
print("[PATCH] Applied patched_torch_ones globally to torch.ones.")

# 5. Squeeze Patch (Global)
# Replace .squeeze() with .reshape() to avoid "ValueError: input_shape[axes] all point to dims==1" in QNN/TFLite
original_tensor_squeeze = torch.Tensor.squeeze
original_torch_squeeze = torch.squeeze


def patched_squeeze(input, dim=None):
    if dim is None:
        new_shape = [d for d in input.shape if d != 1]
    else:
        # dim can be int or tuple
        dims_to_squeeze = [dim] if isinstance(dim, int) else dim
        # Handle negative dims
        dims_to_squeeze = [d if d >= 0 else d + input.ndim for d in dims_to_squeeze]
        new_shape = [d for i, d in enumerate(input.shape) if i not in dims_to_squeeze]

    return input.reshape(new_shape)


torch.squeeze = patched_squeeze
torch.Tensor.squeeze = patched_squeeze
print("[PATCH] Applied patched_squeeze globally (redirects to reshape).")

# 6. Unbind Patch (Global)
# Replace .unbind() with split+squeeze(patched) to ensure Squeeze ops are redirected to Reshape
original_torch_unbind = torch.unbind
original_tensor_unbind = torch.Tensor.unbind


def patched_unbind(input, dim=0):
    # Prepare chunks using split (which keeps the dimension as size 1)
    chunks = input.split(1, dim=dim)
    # Explicitly call .squeeze(dim) on each chunk.
    # Since we patched .squeeze, this will use .reshape, avoiding Squeeze op in ONNX.
    return tuple(c.squeeze(dim) for c in chunks)


torch.unbind = patched_unbind
torch.Tensor.unbind = patched_unbind
print("[PATCH] Applied patched_unbind globally (redirects to split+reshape).")

# 7. Transformers Docstring Patch (Fix for TypeError in some envs)
import transformers.utils


def no_op_decorator(*args, **kwargs):
    # Case 1: Called as @auto_docstring (args[0] is function)
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return args[0]

    # Case 2: Called with args (factory) -> return identity decorator
    def intermediate_decorator(func):
        return func

    return intermediate_decorator


transformers.utils.auto_docstring = no_op_decorator
print("[PATCH] Disabled transformers.utils.auto_docstring (Robust).")

from transformers import AutoConfig, AutoModel, AutoTokenizer
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration

# 1. Aggressive RoPE Patch (Fixes KeyError: 'default')
try:
    from qwen_tts.core.models.modeling_qwen3_tts import ROPE_INIT_FUNCTIONS

    if "linear" in ROPE_INIT_FUNCTIONS:
        original_linear = ROPE_INIT_FUNCTIONS["linear"]

        def patched_default_init(config, device):
            if not hasattr(config, "rope_scaling") or not isinstance(
                config.rope_scaling, dict
            ):
                config.rope_scaling = {"type": "linear", "factor": 1.0}
            if "factor" not in config.rope_scaling:
                config.rope_scaling["factor"] = 1.0
            if "type" not in config.rope_scaling:
                config.rope_scaling["type"] = "linear"
            return original_linear(config, device)

        if "default" not in ROPE_INIT_FUNCTIONS:
            ROPE_INIT_FUNCTIONS["default"] = patched_default_init
except ImportError:
    pass

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
PROJECT_NAME = "qwen3_tts_12hz_0_6b_customvoice"


def _get_attr(obj, name):
    return getattr(obj, name, None)


class TalkerModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask):
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.long(),
            return_dict=False,
        )[0]
        # Force attention_mask into graph to prevent ONNX pruning
        out = out + (attention_mask.sum() * 0.0)
        return out


class CodePredictorWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask):
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.long(),
            return_dict=False,
        )[0]


class SpeechDecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio_codes):
        # model.decode always returns a tuple or object. We need raw tensor.
        # It's safest to return_dict=False
        out = self.model.decode(audio_codes, return_dict=False)

        # Unwrap tuple if present (customary for HF models)
        if isinstance(out, tuple):
            out = out[0]

        # Unwrap LIST if present (Decoder returns list of tensors for batch, we want single tensor for trace)
        if isinstance(out, list):
            out = out[0]

        return out


class EmbeddingWrapper(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.e = e

    def forward(self, x):
        return self.e(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "talker", "cp", "decoder"],
        help="Submodule to trace",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Run parity checks after tracing"
    )
    args = parser.parse_args()

    print(f"[INFO] Starting Trace for {MODEL_ID}")

    # Register Checks
    # Patch AutoModel.register to be robust against "config_class is None" errors from broken qwen_tts code
    original_register = AutoModel.register

    def safe_register(config_class, model_class, exist_ok=False):
        try:
            # Fix the None config_class issue dynamically
            if getattr(model_class, "config_class", None) is None:
                model_class.config_class = config_class
            original_register(config_class, model_class, exist_ok=exist_ok)
        except ValueError as e:
            if "config_class" in str(e):
                print(f"[WARN] Suppressed AutoModel.register config mismatch: {e}")
            else:
                raise e

    AutoModel.register = safe_register

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    # Config
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    if hasattr(config, "talker_config"):
        tc = config.talker_config
        if not hasattr(tc, "pad_token_id"):
            tc.pad_token_id = 151936

    # CP Config copy if missing
    import copy

    if not hasattr(config, "code_predictor_config"):
        config.code_predictor_config = copy.deepcopy(config.talker_config)

    # Force Eager Attention (Bypasses SDPA export issues)
    config._attn_implementation = "eager"
    if hasattr(config, "talker_config"):
        config.talker_config._attn_implementation = "eager"
    if hasattr(config, "decoder_config"):
        config.decoder_config._attn_implementation = "eager"
    if hasattr(config, "code_predictor_config"):
        config.code_predictor_config._attn_implementation = "eager"

    # Load Model
    model = AutoModel.from_pretrained(
        MODEL_ID, config=config, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()

    inputs_dir = f"model_zoo/{PROJECT_NAME}/inputs"
    model_dir = f"model_zoo/{PROJECT_NAME}/model"
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print(f"[DEBUG] inputs_dir absolute path: {os.path.abspath(inputs_dir)}")
    print(f"[DEBUG] model_dir absolute path: {os.path.abspath(model_dir)}")

    seq_len = 512
    codec_len = 128

    # [FIX] Manually load ALL weights from safetensors to ensure correct initialization
    # AutoModel.from_pretrained seems to fail silently for some sub-modules (text_projection, embeddings)
    print("\n[INFO] Manually loading Model weights (Global Fix)...")
    try:
        from transformers.utils.hub import cached_file
        from safetensors.torch import load_file as load_safetensors

        # 1. Main Model
        st_file = cached_file(MODEL_ID, "model.safetensors")
        if st_file:
            print(f"[INFO] Found checkpoint: {st_file}")
            st_weights = load_safetensors(st_file)

            # The model variable is Qwen3TTSForConditionalGeneration
            # It matches the keys in safetensors mostly exactly
            missing, unexpected = model.load_state_dict(st_weights, strict=False)
            print(
                f"[INFO] Global Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )

            # Filter out expected missing keys (like speech_tokenizer which is separate)
            real_missing = [m for m in missing if not m.startswith("speech_tokenizer")]
            if real_missing:
                print(f"[WARN] Real Missing keys: {len(real_missing)}")
                for m in real_missing[:5]:
                    print(f"  - {m}")
        else:
            print("[WARN] Could not find model.safetensors")

    except Exception as e:
        print(f"[ERROR] Failed to fix weights: {e}")
        import traceback

        traceback.print_exc()

    talker = model.talker
    talker_model = talker.model
    code_predictor = getattr(talker, "code_predictor", None)
    speech_tokenizer = getattr(
        model, "speech_tokenizer", getattr(model, "decoder", None)
    )

    # [FIX] Manually load speech tokenizer weights from safetensors to ensure correct initialization
    # (AutoModel loading resulted in random weights due to likely file/key mismatch in cache)
    if speech_tokenizer and hasattr(speech_tokenizer, "model"):
        st_model = speech_tokenizer.model
        print("\n[INFO] Manually loading Speech Tokenizer weights into nn.Module...")
        try:
            from transformers.utils.hub import cached_file
            from safetensors.torch import load_file as load_safetensors

            st_file = cached_file(MODEL_ID, "speech_tokenizer/model.safetensors")
            if st_file:
                print(f"[INFO] Found checkpoint: {st_file}")
                st_weights = load_safetensors(st_file)

                # keys in safetensors start with "decoder...", matching the model structure
                missing, unexpected = st_model.load_state_dict(st_weights, strict=False)
                print(
                    f"[INFO] Weights Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
                )

                # Verify critical keys
                if len(missing) > 0:
                    real_missing = [m for m in missing if "beta" in m or "gamma" in m]
                    if real_missing:
                        print(
                            f"[WARN] Critical Missing keys (beta/gamma): {real_missing[:5]}..."
                        )
            else:
                print("[WARN] Could not find speech_tokenizer/model.safetensors")
        except Exception as e:
            print(f"[ERROR] Failed to fix weights: {e}")
            import traceback

            traceback.print_exc()
            # Failsafe: Continue but warn heavily

    # CRITICAL: Verify Reference Model Output immediately after loading
    # This prevents spending time tracing a broken model.
    print("[INFO] Verifying Reference Model Sanity...")
    try:
        # Talker check
        sanity_ids = torch.randint(0, 1000, (1, 10), dtype=torch.long)
        sanity_emb = talker.model.text_embedding(sanity_ids)
        if sanity_emb.abs().mean() == 0:
            print(
                "[ERROR] Reference Model Text Embedding is ALL SCALAR ZEROS! Weights not loaded!"
            )
            sys.exit(1)
        else:
            print(
                f"[PASS] Reference Model Text Embedding Sanity (Mean: {sanity_emb.abs().mean():.4f})"
            )
    except Exception as e:
        print(f"[WARN] Failed sanity check: {e}")

    seq_len = 512
    codec_len = 128

    # --- GLOBAL PATCHES ---
    # (Applied at top level)

    # --- TALKER ---
    if args.target in ["all", "talker"]:
        print("\n[INFO] Tracing Talker...")

        # 1. Embeddings
        emb_path = os.path.join(model_dir, "qwen3_tts_text_embedding.pt")
        text_emb_layer = getattr(
            talker_model, "embed_tokens", getattr(talker_model, "text_embedding", None)
        )

        if text_emb_layer is None:
            print("[ERROR] Could not find text embedding layer.")
            sys.exit(1)

        # Save Text Embedding
        # Save Text Embedding

        try:
            dummy_idx = torch.zeros((1, 10), dtype=torch.long)
            traced_emb = torch.jit.trace(
                EmbeddingWrapper(text_emb_layer).eval(), dummy_idx
            )
            torch.jit.save(traced_emb, emb_path)
            print(f"[INFO] Saved text embedding: {emb_path}")
        except Exception as e:
            print(f"[WARN] Failed text emb: {e}")

        # 2. Main Talker
        # Use 128 to match verification
        trace_seq_len = 128
        input_ids = torch.randint(0, 151936, (1, trace_seq_len), dtype=torch.long)

        # Use proper embedding lookup (handle module structure)
        if hasattr(talker, "text_embedding"):
            text_emb = talker.text_embedding(input_ids)
        else:
            text_emb = talker.get_text_embeddings()(input_ids)

        inputs_embeds = talker.text_projection(text_emb)
        attn_mask = torch.ones((1, trace_seq_len), dtype=torch.int32)

        np.save(
            f"{inputs_dir}/talker_inputs_embeds.npy", inputs_embeds.detach().numpy()
        )
        np.save(f"{inputs_dir}/talker_attention_mask.npy", attn_mask.numpy())

        wrapper = TalkerModelWrapper(talker_model).eval()
        traced_talker = torch.jit.trace(
            wrapper, (inputs_embeds, attn_mask), strict=False, check_trace=False
        )
        torch.jit.save(traced_talker, f"{model_dir}/qwen3_tts_talker_model.pt")
        print(f"[INFO] Saved talker model.")

        # Revert Patch (Optional, but good practice)
        F.scaled_dot_product_attention = original_sdpa

        # 3. Projections
        # Text Proj
        # Trace projection layer: Input 2048 -> Output 1024 (or whatever it is)
        tp = talker.text_projection
        traced_proj = torch.jit.trace(tp.eval(), text_emb)
        torch.jit.save(traced_proj, f"{model_dir}/qwen3_tts_text_projection.pt")
        np.save(f"{inputs_dir}/text_projection_input.npy", text_emb.detach().numpy())
        print("[INFO] Saved text projection.")

        # Codec Head
        ch = talker.codec_head
        ch_in = torch.randn(1, codec_len, talker.config.hidden_size)  # 1024?
        traced_ch = torch.jit.trace(ch.eval(), ch_in)
        torch.jit.save(traced_ch, f"{model_dir}/qwen3_tts_codec_head.pt")
        np.save(f"{inputs_dir}/codec_head_input.npy", ch_in.detach().numpy())
        print("[INFO] Saved codec head.")

    # --- CP ---
    if args.target in ["all", "cp"] and code_predictor:
        print("\n[INFO] Tracing Code Predictor...")
        cp_model = code_predictor.model
        # Codec Embedding
        cp_emb_path = f"{model_dir}/qwen3_tts_codec_embedding.pt"
        cp_emb_layer = getattr(
            cp_model,
            "embed_tokens",
            getattr(
                cp_model, "text_embedding", getattr(cp_model, "codec_embedding", None)
            ),
        )

        if cp_emb_layer:
            # Handle ModuleList (likely multiple codebooks)
            if isinstance(cp_emb_layer, nn.ModuleList):
                print(
                    f"[INFO] CP Embedding is ModuleList (len={len(cp_emb_layer)}). Tracing index 0."
                )
                cp_emb_layer = cp_emb_layer[0]

            traced_cp_emb = torch.jit.trace(
                EmbeddingWrapper(cp_emb_layer).eval(),
                torch.zeros((1, 10), dtype=torch.long),
            )
            torch.jit.save(traced_cp_emb, cp_emb_path)
            print("[INFO] Saved codec embedding.")
        else:
            print(
                "[WARN] Could not find CP embedding layer. Skipping embedding trace (might be shared?)."
            )

        # CP Body
        cp_hidden = cp_model.config.hidden_size
        cp_in = torch.randn(1, codec_len, cp_hidden)
        cp_mask = torch.ones((1, codec_len), dtype=torch.int32)

        np.save(
            f"{inputs_dir}/code_predictor_inputs_embeds.npy", cp_in.detach().numpy()
        )
        np.save(f"{inputs_dir}/code_predictor_attention_mask.npy", cp_mask.numpy())

        wrapper = CodePredictorWrapper(cp_model).eval()
        traced_cp = torch.jit.trace(
            wrapper, (cp_in, cp_mask), strict=False, check_trace=False
        )
        torch.jit.save(traced_cp, f"{model_dir}/qwen3_tts_code_predictor.pt")
        print("[INFO] Saved CP model.")

    # --- DECODER ---
    if args.target in ["all", "decoder"] and speech_tokenizer:
        print("\n[INFO] Tracing Decoder...")

        # PATCH: create_causal_mask to avoid Boolean Not op AND Opset 11 tril/triu issues
        # Use manual index comparison and Where
        def patched_create_causal_mask(
            input_embeds, attention_mask=None, past_key_values=None, **kwargs
        ):
            bsz, seq_len = input_embeds.shape[:2]
            dtype = input_embeds.dtype
            device = input_embeds.device

            # Create indices [seq_len, 1] and [1, seq_len]
            rows = torch.arange(seq_len, device=device).view(-1, 1)
            cols = torch.arange(seq_len, device=device).view(1, -1)

            # Upper Triangle (col > row) is masked (-inf)
            # Lower Triangle (col <= row) is kept (0.0)
            is_upper = cols > rows  # Bool tensor

            # Use -1e9 instead of -inf for robustness against NaN in Softmax
            min_val = torch.tensor(-1e9, device=device, dtype=dtype)
            zero_val = torch.tensor(0.0, device=device, dtype=dtype)

            # where(condition, x, y)
            mask = torch.where(is_upper, min_val, zero_val)

            return mask.view(1, 1, seq_len, seq_len).expand(bsz, 1, seq_len, seq_len)

        try:
            from qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2

            modeling_qwen3_tts_tokenizer_v2.create_causal_mask = (
                patched_create_causal_mask
            )
            print("[PATCH] Applied numeric patched_create_causal_mask (No Boolean Ops)")

            # PATCH: RotaryEmbedding with Broadcasting to avoid QNN Segfault (Static Shape Inference)
            OriginalRotary = modeling_qwen3_tts_tokenizer_v2.Qwen3TTSTokenizerV2DecoderRotatoryEmbedding

            class PatchedRotaryEmbedding(OriginalRotary):
                def forward(self, x, position_ids):
                    # Simplified Broadcasting implementation
                    # input: x [bsz, seq_len, heads, head_dim] (or similar, not used for shape here)
                    # position_ids: [bsz, seq_len]
                    # inv_freq: [head_dim/2]

                    # Force float32 for precision
                    inv_freq = self.inv_freq.float().to(x.device)
                    position_ids = position_ids.float()

                    # [bsz, seq_len, 1]
                    pos_expanded = position_ids.unsqueeze(-1)
                    # [1, 1, head_dim/2]
                    freq_expanded = inv_freq.view(1, 1, -1)

                    # Broadcast Mul: [bsz, seq_len, head_dim/2]
                    freqs = pos_expanded * freq_expanded

                    # Concatenate to get full head_dim
                    emb = torch.cat((freqs, freqs), dim=-1)

                    cos = emb.cos() * self.attention_scaling
                    sin = emb.sin() * self.attention_scaling

                    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

            modeling_qwen3_tts_tokenizer_v2.Qwen3TTSTokenizerV2DecoderRotatoryEmbedding = PatchedRotaryEmbedding
            print(
                "[PATCH] Applied PatchedRotaryEmbedding (Broadcasting instead of MatMul)"
            )

        except ImportError:
            print(
                "[WARN] Could not patch create_causal_mask or RotaryEmbedding. QNN Conversion might fail."
            )

        # Diff Patch
        def robust_diff(input, n=1, dim=-1, prepend=None, append=None):
            # print(f"[DEBUG] robust_diff called. Input: {input.shape} {input.dtype}") # Reduce spam

            # ONNX 'Sub' op does not support Boolean inputs (PyTorch implicit cast to int works, but traces to invalid ONNX)
            if input.dtype == torch.bool:
                input = input.to(torch.int32)
                if prepend is not None:
                    prepend = prepend.to(torch.int32)
                if append is not None:
                    append = append.to(torch.int32)

            if n != 1:
                raise ValueError("robust_diff only supports n=1")

            # Resolve dim to positive index
            ndim = input.ndim
            if dim < 0:
                dim += ndim

            if prepend is None and append is None:
                L = input.shape[dim]
                # Slice 1: 1 to End -> narrow(dim, 1, L-1)
                # Slice 2: 0 to End-1 -> narrow(dim, 0, L-1)
                res = input.narrow(dim, 1, L - 1) - input.narrow(dim, 0, L - 1)
                print(f"[DEBUG] robust_diff (standard): Result shape {res.shape}")
                return res

            parts = []
            if prepend is not None:
                parts.append(prepend)
            parts.append(input)
            if append is not None:
                parts.append(append)

            combined = torch.cat(parts, dim=dim)

            # Optimization for Prepend-only case (Common)
            if prepend is not None and append is None:
                # Use split instead of narrow to generate Split op instead of Slice op (safer for QNN)
                input_len = input.shape[dim]
                # We expect combined to have length input_len + 1.
                # Split into [input_len, 1]
                sliced_prev, _ = torch.split(combined, [input_len, 1], dim=dim)
                res = input - sliced_prev
                print(
                    f"[DEBUG] robust_diff (prepend-opt-split): Result shape {res.shape}"
                )
                return res

            L = combined.shape[dim]
            # Use split for generic case too if possible, but fallback to narrow/slice logic here for now as it's less common
            res = combined.narrow(dim, 1, L - 1) - combined.narrow(dim, 0, L - 1)
            print(f"[DEBUG] robust_diff (padded-generic): Result shape {res.shape}")
            return res

        torch.diff = robust_diff
        print("[PATCH] Applied robust_diff")

        tokenizer_model = speech_tokenizer.model
        num_q = getattr(config.talker_config, "num_code_groups", 8)
        dummy_codes = torch.randint(0, 1024, (1, codec_len, num_q), dtype=torch.long)

        wrapper = SpeechDecoderWrapper(tokenizer_model).eval()
        tokenizer_model.to("cpu")
        traced_decoder = torch.jit.trace(wrapper, (dummy_codes,), strict=False)
        torch.jit.save(traced_decoder, f"{model_dir}/qwen3_tts_speech_decoder.pt")
        np.save(f"{inputs_dir}/speech_decoder_audio_codes.npy", dummy_codes.numpy())
        print("[INFO] Saved decoder.")

    print("\n[INFO] TRACE COMPLETE")

    if args.verify:
        print("\n[INFO] Verifying Parity (Original vs Traced)...")
        # 1. Verify Talker
        if args.target in ["all", "talker"]:
            print("Verifying Talker...")
            with torch.no_grad():
                # Run Original (Full Qwen2 Model)
                # Note: We must replicate exactly what the wrapper does.
                # Wrapper takes (inputs_embeds, attention_mask) calls model(..., return_dict=False)[0]

                # We need fresh inputs to be safe
                v_input_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
                v_inputs_embeds = talker.get_text_embeddings()(v_input_ids)
                v_inputs_embeds = talker.text_projection(v_inputs_embeds)
                v_mask = torch.ones((1, seq_len), dtype=torch.int32)

                # Original
                orig_out = talker_model(
                    inputs_embeds=v_inputs_embeds,
                    attention_mask=v_mask.long(),
                    return_dict=False,
                )[0]

                # Traced
                traced_talker_loaded = torch.jit.load(
                    f"{model_dir}/qwen3_tts_talker_model.pt"
                )
                traced_out = traced_talker_loaded(v_inputs_embeds, v_mask)

                # Compare
                diff = (orig_out - traced_out).abs().max().item()
                if diff < 1e-4:
                    print(f"[PASS] Talker Parity Confirmed (Max Diff: {diff:.6f})")
                else:
                    print(f"[FAIL] Talker Parity Mismatch (Max Diff: {diff:.6f})")

        # 2. Verify Code Predictor
        if args.target in ["all", "cp"] and code_predictor:
            print("Verifying Code Predictor...")
            with torch.no_grad():
                v_cp_in = torch.randn(
                    1, codec_len, code_predictor.model.config.hidden_size
                )
                v_cp_mask = torch.ones((1, codec_len), dtype=torch.int32)

                # Original
                orig_cp_out = code_predictor.model(
                    inputs_embeds=v_cp_in,
                    attention_mask=v_cp_mask.long(),
                    return_dict=False,
                )[0]

                # Traced
                traced_cp_loaded = torch.jit.load(
                    f"{model_dir}/qwen3_tts_code_predictor.pt"
                )
                traced_cp_out = traced_cp_loaded(v_cp_in, v_cp_mask)

                diff = (orig_cp_out - traced_cp_out).abs().max().item()
                if diff < 1e-4:
                    print(f"[PASS] CP Parity Confirmed (Max Diff: {diff:.6f})")
                else:
                    print(f"[FAIL] CP Parity Mismatch (Max Diff: {diff:.6f})")


if __name__ == "__main__":
    main()
