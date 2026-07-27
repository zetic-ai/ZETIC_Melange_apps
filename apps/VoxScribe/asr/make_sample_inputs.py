#!/usr/bin/env python3
# Regenerates the sample_input_*.npy files (shape/dtype only — Melange needs no
# real values). Run: python make_sample_inputs.py  (numpy is the only dep.)
# These match the SERVED signatures of OpenAI/whisper-tiny-{encoder,decoder};
# see model_selection.md / spec_stub_asr.md for sources.
import numpy as np

# Encoder: log-mel [1,80,3000] float32
np.save("sample_input_encoder.npy",
        np.random.rand(1, 80, 3000).astype(np.float32))

# Decoder (no KV-cache, fixed length 448):
ids = np.full((1, 448), 50256, dtype=np.int32); ids[0, 0] = 50258  # pad + SOT
np.save("sample_input_decoder_input_ids.npy", ids)
np.save("sample_input_decoder_encoder_hidden_states.npy",
        np.random.rand(1, 1500, 384).astype(np.float32))
mask = np.zeros((1, 448), dtype=np.int32); mask[0, 0] = 1
np.save("sample_input_decoder_attention_mask.npy", mask)
print("wrote 4 sample_input_*.npy")
