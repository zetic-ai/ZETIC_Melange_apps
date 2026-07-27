#!/usr/bin/env python3
# =============================================================================
# VoxScribe ASR — Whisper-tiny encoder + decoder ONNX export recipe
# =============================================================================
#
# STATUS: NOT EXECUTED in the Explorer environment.
#   The sandbox had no working torch/transformers/onnx (the only usable
#   interpreter was the macOS system Python 3.9 with numpy 2.0.2; the Homebrew
#   Python 3.14 is broken — pyexpat symbol error — and there is no torch).
#   This file is therefore a complete, re-runnable RECIPE, written to match the
#   tensor signatures actually used by the shipping app in apps/whisper-tiny/
#   (iOS WhisperEncoder.swift / WhisperDecoder.swift, Android WhisperEncoder.kt
#   / WhisperDecoder.kt). Run it on a machine with the deps below to regenerate
#   the ONNX files if a fresh re-registration is ever needed.
#
# IMPORTANT — for VoxScribe we REUSE the already-registered Melange models:
#       OpenAI/whisper-tiny-encoder   (version 1)
#       OpenAI/whisper-tiny-decoder   (version 1)
#   so producing new ONNX is OPTIONAL. See melange_upload.md. This recipe exists
#   for reproducibility / auditability and to document the exact static shapes.
#
# Deps (a known-good combo):
#   python -m pip install "torch==2.2.*" "transformers==4.40.*" "onnx>=1.15" "numpy<2"
#
# Run:
#   python export.py
# Produces:
#   whisper-encoder.onnx   input_features[1,80,3000] -> last_hidden_state[1,1500,384]
#   whisper-decoder.onnx   (input_ids[1,448], encoder_hidden_states[1,1500,384],
#                           attention_mask[1,448]) -> logits[1,448,51865]
#
# Discipline (binding — see EXPLORATION.md §5/§7):
#   - STATIC shapes only. dynamic_axes is left None / OFF.
#   - opset 14 (Whisper's decoder uses ops — masked-fill / where / scaled-dot
#     attention — that do not all lower cleanly at opset 12; 12 is the repo
#     reference for YOLO, but Whisper needs 14. Stated explicitly per the doc's
#     "opset ~12" guidance.)
#   - half=False. Keep the ONNX in float32; Melange decides precision server-side.
#   - NO KV-cache / past_key_values. The decoder is exported as a fixed-length
#     [1,448] full-sequence recompute, exactly as the app drives it (it refills
#     the 448-slot id buffer each step and reads the logits row at the current
#     position). This is the simplest static-shape decoder for Melange.
# =============================================================================

import torch
from transformers import WhisperForConditionalGeneration

MODEL_ID   = "openai/whisper-tiny"   # MIT license
DEC_LEN    = 448                     # n_text_ctx — fixed decoder sequence length
N_MELS     = 80                      # log-mel bins
N_FRAMES   = 3000                    # 30s @ 16kHz, hop 160  -> 3000 frames
N_AUDIO_CTX= 1500                    # encoder output time steps
D_MODEL    = 384                     # whisper-tiny hidden size
OPSET      = 14

# Token IDs (whisper-tiny multilingual tokenizer), for reference / sample input:
SOT = 50258   # <|startoftranscript|>
EOT = 50257   # <|endoftext|>  (decode-loop termination token)
PAD = 50256   # filler for unused decoder slots (masked out by attention_mask)


def main():
    print(f"Loading {MODEL_ID} ...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()

    # -------------------------------------------------------------------------
    # 1) ENCODER:  input_features[1,80,3000] -> last_hidden_state[1,1500,384]
    # -------------------------------------------------------------------------
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.encoder = m.model.encoder
        def forward(self, input_features):
            return self.encoder(input_features).last_hidden_state

    enc = EncoderWrapper(model).eval()
    dummy_mel = torch.randn(1, N_MELS, N_FRAMES, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            enc, (dummy_mel,), "whisper-encoder.onnx",
            input_names=["input_features"],
            output_names=["last_hidden_state"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,          # STATIC — no dynamic axes
        )
    print("wrote whisper-encoder.onnx")

    # -------------------------------------------------------------------------
    # 2) DECODER (no KV-cache, fixed length 448):
    #    (input_ids[1,448] int32, encoder_hidden_states[1,1500,384] float32,
    #     attention_mask[1,448] int32) -> logits[1,448,51865] float32
    #
    #    NB the shipping app sends int32 token/mask buffers (Kotlin IntArray =
    #    4 bytes; iOS Int32). The registered model therefore takes int32, not the
    #    int64 that optimum emits by default. We cast inside the wrapper so the
    #    exported graph's declared input dtype is int32, matching the app.
    # -------------------------------------------------------------------------
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.decoder  = m.model.decoder
            self.proj_out = m.proj_out          # lm_head -> vocab 51865
        def forward(self, input_ids, encoder_hidden_states, attention_mask):
            out = self.decoder(
                input_ids=input_ids.to(torch.long),
                attention_mask=attention_mask.to(torch.long),
                encoder_hidden_states=encoder_hidden_states,
                use_cache=False,
            )
            return self.proj_out(out.last_hidden_state)

    dec = DecoderWrapper(model).eval()
    dummy_ids  = torch.full((1, DEC_LEN), PAD, dtype=torch.int32)
    dummy_ids[0, 0] = SOT
    dummy_hs   = torch.randn(1, N_AUDIO_CTX, D_MODEL, dtype=torch.float32)
    dummy_mask = torch.zeros((1, DEC_LEN), dtype=torch.int32)
    dummy_mask[0, 0] = 1
    with torch.no_grad():
        torch.onnx.export(
            dec, (dummy_ids, dummy_hs, dummy_mask), "whisper-decoder.onnx",
            input_names=["input_ids", "encoder_hidden_states", "attention_mask"],
            output_names=["logits"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,          # STATIC — no dynamic axes
        )
    print("wrote whisper-decoder.onnx")

    # -------------------------------------------------------------------------
    # 3) Confirm NO dynamic axes (every dim must be a fixed integer).
    # -------------------------------------------------------------------------
    import onnx
    for path in ("whisper-encoder.onnx", "whisper-decoder.onnx"):
        g = onnx.load(path).graph
        for vi in list(g.input) + list(g.output):
            dims = vi.type.tensor_type.shape.dim
            shape = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in dims]
            assert all(isinstance(s, int) and s > 0 for s in shape), \
                f"DYNAMIC AXIS in {path}: {vi.name} -> {shape}"
            print(f"{path}: {vi.name} {shape}")
    print("OK — all axes static.")


if __name__ == "__main__":
    main()
