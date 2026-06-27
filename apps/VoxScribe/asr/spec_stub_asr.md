# SPEC (stub): VoxScribe — ASR model

> Pre-drafted by the Explorer (Stage 0). Everything the architecture reveals is
> filled in. **GATE-0 fields (Melange name/version, SERVED shapes, modelMode)
> are left blank** for the human to confirm via `melange_upload.md`. This is the
> ASR half only; diarization is a separate model/spec owned by the companion
> Explorer. The worker fuses ASR text with diarization speaker turns downstream.

## One-line pitch
On-device, speaker-labeled live transcript ("who spoke when") for prospect
Kardome — Whisper-tiny does the speech-to-text; a diarizer supplies speaker turns.

## Model
- Source (HF repo / origin): openai/whisper-tiny (MIT)
- Architecture: Whisper-tiny seq2seq, exported as a **two-model split** (encoder
  + decoder), decoder **without KV-cache**, fixed decoder length 448.
- Melange model name (encoder): **OpenAI/whisper-tiny-encoder**  _(reuse — confirm at GATE 0)_
- Melange model name (decoder): **OpenAI/whisper-tiny-decoder**  _(reuse — confirm at GATE 0)_
- Melange version: **[GATE 0 — confirm, expected v1]**
- **Encoder input tensor**: float32 `[1, 80, 3000]`, layout `[batch, n_mels, frames]`
  (NOT image NCHW). Values = Whisper log-mel (see pre-processing); no extra scaling.
- **Encoder output tensor**: float32 `[1, 1500, 384]` = encoder hidden states
  (n_audio_ctx 1500, d_model 384).
- **Decoder input tensors**:
  1. `input_ids` int32 `[1, 448]` — token ids; slot 0 = SOT (50258), unused slots
     filled with pad 50256, masked off.
  2. `encoder_hidden_states` float32 `[1, 1500, 384]` — the encoder output, passed
     unchanged every decode step.
  3. `attention_mask` int32 `[1, 448]` — 1 for valid token positions, 0 for pad.
- **Decoder output tensor**: float32 `[1, 448, 51865]` = per-position logits over
  vocab 51865. Read the row at the current decode index.
- **SERVED shapes (GATE 0)**: encoder __________ / decoder __________  _(paste back)_
- **Post-processing baked into ONNX?** No. No NMS (N/A for ASR). Argmax/greedy
  decode, EOT termination, and token->text detokenization are all client-side.
- Classes / labels: N/A (vocabulary, not classes). Vocab size 51865; tokenizer
  vocab.json ships with the app (see `apps/whisper-tiny/.../vocab.json`).
  Special tokens: SOT `<|startoftranscript|>`=50258, EOT `<|endoftext|>`=50257,
  pad=50256.
- modelMode to use and why: **RUN_AUTO** _(confirm at GATE 0)_. No client mode
  steers backend selection or avoids the iOS-26 GPU crash (that's server-side,
  ZETIC). Read the served artifact from the device console as ground truth.

## Input source
- Microphone, live capture.
- Sample rate / format requested: **16 kHz, mono, float PCM** (Whisper's required
  rate; Android `AudioSampler` uses `SAMPLE_RATE=16000`, `CHANNEL_IN_MONO`,
  `ENCODING_PCM_FLOAT`). Any other capture rate MUST be resampled to 16 kHz.
- Orientation handling: N/A (audio).

## Pre-processing pipeline (ordered, exact)
1. Capture mic audio; ensure **16 kHz mono** float PCM (resample if the device
   delivers 44.1/48 kHz; downmix to mono if stereo).
2. Window to a fixed **30 s** span: pad with zeros or truncate so the sample count
   is exactly 30 s × 16 kHz = 480000 samples.
3. Compute the **log-mel spectrogram**: STFT `n_fft=400`, `hop=160`, Hann window;
   `n_mels=80`; magnitude -> mel filterbank -> `log10`; Whisper clamp+scale
   (`log_spec = max(log_spec, log_spec.max() - 8.0)`, then `(log_spec + 4.0) / 4.0`).
   Result `[80, 3000]` -> add batch -> `[1, 80, 3000]` float32.
   - NOTE: in the shipping app this is done by the SDK's `WhisperWrapper.process()`,
     not hand-rolled. The VoxScribe worker either (a) reuses an equivalent
     Dart/native log-mel that reproduces these exact params, or (b) calls a wrapper.
     The 80/400/160/3000 numbers above are the contract the encoder expects.
4. Feed `[1,80,3000]` to the encoder; cache its `[1,1500,384]` output.

## Post-processing pipeline (ordered, exact) — the greedy decode loop (client-side)
1. Init `input_ids[1,448]` filled with pad 50256; `input_ids[0]=50258` (SOT).
   Init `attention_mask[1,448]=0`; `attention_mask[0]=1`. `idx=1`.
   (For multilingual whisper-tiny you may also seed language + `<|transcribe|>` +
   `<|notimestamps|>` tokens after SOT; the reference app seeds only SOT.)
2. Loop while `idx < 448`:
   a. Run decoder with (`input_ids`, cached `encoder_hidden_states`, `attention_mask`).
   b. Slice logits row at position `idx-1`: `logits[(idx-1)*51865 : idx*51865]`.
   c. `next = argmax(row)`.
   d. If `next == EOT (50257)`: **stop**.
   e. `input_ids[idx]=next`; `attention_mask[idx]=1`; append `next`; `idx++`.
3. Detokenize the collected ids via vocab.json (skip special tokens) -> transcript
   text. Hand text + timing to the fusion step for speaker labeling.

> Dtype trap: token/mask tensors are **int32** in the shipping clients (4-byte
> ints). Build them as int32 in Dart unless GATE 0 reports the served model wants
> int64.

## UI
- Left to the worker. Functional must-haves: live scrolling transcript, per-segment
  speaker label (from diarization fusion), and an inference-latency / RTF readout.

## Platform targets
- iOS minimum: per repo (iOS 16.6+); Android minSdk 24.
- Known traps for this model/artifact:
  - The iOS/macOS 26.3+ CoreML-GPU MPSGraph crash is possible for any
    attention-heavy graph; it's handled server-side by ZETIC (GPU filtered).
    Confirm the served artifact is not GPU on affected OS via the device console.
  - "Benchmarked != served": don't assume the fast NPU row is what runs; budget
    for a CPU-speed fallback until `runtimeApType=NPU` is confirmed on-device.
  - Two model loads (encoder + decoder) + a 448-step decode loop = watch cold-start
    and per-utterance latency; the encoder runs once, the decoder many times.

## Validation focus (AUDIO-domain correctness traps — replace the vision traps)
- **Sample-rate mismatch**: Whisper REQUIRES 16 kHz. Capturing at 44.1/48 kHz and
  feeding it unresampled produces garbage (pitch/time distortion). Test the
  resampler.
- **Mono vs stereo**: must downmix to a single channel before mel; test a stereo
  input collapses correctly.
- **Log-mel params exactness**: `n_fft=400`, `hop=160`, `n_mels=80`, 30 s window
  padded/truncated to **exactly 3000 frames**. Off-by-one framing or wrong hop ->
  shape `[1,80,2999/3001]` or shifted features. Unit-test frame count == 3000.
- **Mel normalization**: Whisper's specific `log10` + clamp to `max-8.0` + `(+4)/4`
  scale. A plain log-mel without the clamp/scale degrades WER badly. Test the
  normalization formula against a known reference vector.
- **Greedy decode termination**: must stop on EOT (50257), not run all 448 steps;
  and must read the logit row at the CURRENT position (`idx-1`), not position 0.
  Test loop terminates and indexing is correct.
- **Fixed vs dynamic decoder length**: the model is fixed [1,448]; never send a
  shorter/longer id buffer. Pad with 50256 and mask. Test buffer is always 448.
- **Token dtype int32**: ids/mask must be int32 (confirm at GATE 0). Sending int64
  to an int32-registered model mismatches byte length.
