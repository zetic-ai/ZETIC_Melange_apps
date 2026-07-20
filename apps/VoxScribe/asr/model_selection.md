# Model selection — VoxScribe (ASR / speech-to-text)

> **Decision is pre-made by the orchestrator: reuse Whisper-tiny** via the
> existing registered Melange models `OpenAI/whisper-tiny-encoder` and
> `OpenAI/whisper-tiny-decoder`. This file documents WHY that is the right ASR
> for an on-device demo and records the export shapes. (The companion Explorer
> owns diarization separately — not covered here.)

## Shortlist (top candidates considered)
| Rank | Model | Params / size | License | Export path | Melange-fit notes | Score |
|------|-------|---------------|---------|-------------|-------------------|-------|
| 1 | **openai/whisper-tiny** | 39M (~enc 8MB + dec 29MB fp32) | **MIT** | HF transformers -> torch.onnx, enc/dec split, static shapes | Proven Melange path (already registered + benchmarked); standard conv+attention ops; fixed mel input; no-KV fixed-length decoder converts cleanly | **WINNER** |
| 2 | openai/whisper-base | 74M | MIT | same recipe | ~2x size/latency for modest WER gain; overkill for a live on-device demo | runner-up |
| 3 | distil-whisper/distil-small.en | 166M | MIT | same family | English-only, larger; faster-than-base but still heavier than tiny; no existing Melange registration | — |
| 4 | Systran/faster-whisper-tiny (CT2) | 39M | MIT | CTranslate2 format, NOT ONNX | wrong export path for Melange (CT2, not ONNX) | disq. |
| 5 | nvidia/parakeet / wav2vec2-style CTC | 100M+ | varies | non-Whisper graph | different architecture, no existing Melange path, larger | — |

## Winner: openai/whisper-tiny (reuse existing registration)
- **Smallest viable real ASR**: 39M params, encoder ~8MB + decoder ~29MB fp32 —
  squarely in the on-device "single-digit to low-tens of MB" target.
- **Proven Melange path**: already registered, converted, and benchmarked as a
  two-model split; the shipping `apps/whisper-tiny/` (iOS + Android) drives it,
  so the exact served signatures are knowable from real code, not guessed.
- **License MIT** — clean for ZETIC's GTM / a prospect demo (Kardome).
- **Encoder/decoder split rationale**: Whisper is seq2seq. Splitting lets the
  encoder run **once per 30s window** (heavy conv+attention) while the decoder
  runs autoregressively. Each half is a standalone static-shape ONNX, which
  converts and benchmarks far more cleanly on Melange than one giant graph with a
  generation loop baked in. The decoder is exported **without KV-cache** at a
  **fixed length of 448** and re-runs the full sequence each step (the app refills
  the 448-slot id buffer and reads the logit row at the current position) — the
  simplest possible static-shape decoder for an NPU compiler.
- **Trade-off vs whisper-base**: base lowers WER somewhat but ~doubles size and
  latency and has no existing registration. For a who-spoke-when live-transcript
  demo, tiny's accuracy is sufficient and its latency margin matters more.

## Export (documented; recipe in export.py)
- Recipe: `export.py` (HF `transformers` + `torch.onnx.export`, encoder & decoder
  exported separately). **NOT executed in the Explorer env** (no torch); shapes
  recovered from the shipping `apps/whisper-tiny/` source — see sources below.
- **Encoder**
  - Input:  float32 `[1, 80, 3000]` — log-mel spectrogram (80 mel bins × 3000
    frames = 30s @ 16kHz, hop 160). No NCHW; it's `[batch, mels, frames]`.
  - Output: float32 `[1, 1500, 384]` — encoder hidden states (n_audio_ctx 1500,
    d_model 384). No post-processing baked in.
- **Decoder** (no KV-cache, fixed length 448)
  - Input: `input_ids` int32 `[1, 448]`, `encoder_hidden_states` float32
    `[1, 1500, 384]`, `attention_mask` int32 `[1, 448]`.
  - Output: `logits` float32 `[1, 448, 51865]` (vocab 51865). Argmax/decoding done
    in client code, not baked in.
- **Opset**: 14 (Whisper decoder attention/masking ops don't all lower at 12; 12
  is the repo's YOLO reference). **half=False**. **dynamic_axes OFF (static)** —
  `export.py` asserts every input/output dim is a fixed positive int before exit.

### Sources for each number
- `[1,80,3000]` mel, `[1,1500,384]` enc hidden, `[1,448]` ids/mask, `[1,448,51865]`
  logits, tokens SOT=50258 / EOT=50257 / pad=50256, vocab 51865, no-KV fixed-length
  recompute: **read directly from** `apps/whisper-tiny/` —
  `Android/.../WhisperDecoder.kt` (448 maxLength, `FloatArray(448*51865)`, 3-input
  `model.run`, fill 50256, argmax row), `WhisperEncoder.kt` (single float tensor in),
  `WhisperFeature.kt` (start 50258 / end 50257, model keys), `AudioSampler.kt`
  (16kHz mono float PCM), and the iOS counterparts `WhisperDecoder.swift` /
  `WhisperEncoder.swift` (Int32 buffers -> int32 dtype confirmation).
- d_model 384 / n_audio_ctx 1500 / n_mels 80 / n_text_ctx 448 cross-checked against
  the standard openai/whisper-tiny architecture (HF config).
