# Melange upload — VoxScribe (ASR half)

> **REUSE, do not re-register.** Whisper-tiny is already on Melange as a
> two-model encoder+decoder split. VoxScribe reuses those exact registered
> models — there is normally **nothing to upload**. Your GATE-0 job is to
> **confirm they are READY** and **paste back the served shapes**.

## Models to reuse (already registered)
- encoder: **OpenAI/whisper-tiny-encoder**  (version 1)
- decoder: **OpenAI/whisper-tiny-decoder**  (version 1)

Dashboard links (from apps/whisper-tiny/README.md):
- https://mlange.zetic.ai/p/OpenAI/whisper-tiny-encoder?from=use-cases
- https://mlange.zetic.ai/p/OpenAI/whisper-tiny-decoder?from=use-cases

## Step 1 — Confirm READY
Open both dashboard pages and confirm each shows status **READY** (CONVERTING ->
OPTIMIZING -> READY already completed). If either is not READY, trigger/await the
benchmark before proceeding.

## Step 2 — Read back and paste the SERVED shapes
For **each** model, copy the input/output tensor shapes + dtypes the dashboard
echoes. These are the only values the pipeline cannot know without you.

Expected (documented from the shipping apps/whisper-tiny/ source — confirm they match):

**OpenAI/whisper-tiny-encoder**
- input:  `input_features` float32 `[1, 80, 3000]`  (log-mel spectrogram)
- output: `last_hidden_state` float32 `[1, 1500, 384]`

**OpenAI/whisper-tiny-decoder**  (no KV-cache, fixed sequence length 448)
- input 1: `input_ids` **int32** `[1, 448]`
- input 2: `encoder_hidden_states` float32 `[1, 1500, 384]`
- input 3: `attention_mask` **int32** `[1, 448]`
- output:  `logits` float32 `[1, 448, 51865]`

> Note on dtypes: the shipping iOS/Android clients send the token/mask buffers as
> 4-byte ints (Kotlin `IntArray` / Swift `Int32`), so the registered decoder takes
> **int32**, not the int64 that a default optimum export emits. Confirm the
> dashboard agrees; if it shows int64, flag it (the worker's Dart preprocessor
> must then send int64).

## Contingency — only if shapes DON'T match
If the served shapes differ from the above (e.g. a different decoder length, a
merged single-model graph, KV-cache inputs, or int64 ids), then re-export and
re-register using the artifacts in this folder:
- model:  `whisper-encoder.onnx` + sample `sample_input_encoder.npy`
- model:  `whisper-decoder.onnx` + samples `sample_input_decoder_input_ids.npy`,
  `sample_input_decoder_encoder_hidden_states.npy`,
  `sample_input_decoder_attention_mask.npy`
- (the .onnx files are produced by running `export.py`; the Explorer environment
  could not run torch, so they are NOT pre-built here — only the recipe + the
  shape-correct sample_input .npy files are.)
- register as: `ajayshah/VoxScribe-whisper-encoder` / `ajayshah/VoxScribe-whisper-decoder`, version 1.

## Paste back to the agent (BLOCKED until you do)
- [ ] both models confirmed **READY** (yes/no)
- [ ] encoder served input + output shapes/dtypes
- [ ] decoder served 3 input shapes/dtypes + output shape/dtype
- [ ] whether decoder ids/mask are int32 or int64
- [ ] modelMode: default **RUN_AUTO**
      (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The
      iOS/macOS 26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering
      the GPU candidate; no client mode avoids it. See CLAUDE.md §5.)
