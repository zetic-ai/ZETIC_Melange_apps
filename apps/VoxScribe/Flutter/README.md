# VoxScribe (Flutter)

On-device, speaker-labeled live transcript demo for prospect **Kardome**. A
3-model ZETIC Melange pipeline runs entirely on the device:

1. **pyannote/segmentation-3.0** (`ajayshah/PyannoteSegmentation` v1) — "who
   spoke when" (powerset, 3 local speakers / 7 classes).
2. **Whisper-tiny encoder** (`OpenAI/whisper-tiny-encoder` v1) — log-mel → hidden.
3. **Whisper-tiny decoder** (`OpenAI/whisper-tiny-decoder` v1) — greedy 448-step
   decode → token ids → text.

Fusion is **diarize-then-transcribe**: each segmentation span is transcribed and
attributed to its speaker by construction. The floor input is a bundled ≤10 s,
2-speaker, 16 kHz mono clip (`assets/demo_2spk.wav`).

## Personal key (required)

The ZETIC personal key is injected at build time and is **never committed**:

```bash
flutter run       --release --dart-define=MLANGE_KEY=<your_zetic_key>
flutter build ios --release --dart-define=MLANGE_KEY=<your_zetic_key>
```

If the key is missing the loading screen fails loudly (no silent failure).

## Run (physical device only — no simulator)

The vendored `ZeticMLange.xcframework` ships a device-only `ios-arm64` slice and
must run in **release** mode (debug hangs on recent iOS). iOS 16.6+,
Android minSdk 24.

```bash
flutter build ios --release --dart-define=MLANGE_KEY=<key>
# then sign & install via Xcode / devicectl (see ../HANDOFF.md)
```

## Bundled assets

| Asset | What | Regenerate |
|---|---|---|
| `assets/demo_2spk.wav` | 2-speaker (overlapping) 16 kHz mono clip | macOS `say` + `tool/` (see ../HANDOFF.md) |
| `assets/vocab.json` | GPT-2 byte-level BPE vocab (Whisper) | copied from `apps/whisper-tiny` |
| `assets/mel_filters_80.bin` | OpenAI 80-mel Slaney filterbank `[80,201]` f32 LE | `python3 tool/gen_mel_filters.py assets/mel_filters_80.bin` |

The log-mel golden test vector is produced by `tool/gen_logmel_golden.py`.

## Tests

```bash
flutter test            # Tier A unit suite (14 traps)
flutter test test/benchmark/hot_path_benchmark.dart   # A4 micro-benchmark
```

The native NPU/CPU `run()` is device-only; per-stage latency and RTF are shown
on the in-app HUD (Dart `print` does not surface on a release device console).
