# VoxScribe — GATE 3 Handoff (ready for device)

> Status: **READY FOR DEVICE** (not "done"). Tier A green, Tier B optimized,
> Tier C surfaced below. The human performs the physical-device run.
> Branch `app/voxscribe`. Worktree `/Users/ajayshah/Desktop/ZETIC/voxscribe-wt`.

## Goal

A fully on-device, speaker-labeled live-transcript demo for Flutter (iOS),
powered by a 3-model ZETIC Melange pipeline: pyannote segmentation
(`ajayshah/PyannoteSegmentation` v1) supplies speaker turns; Whisper-tiny
encoder + decoder (`OpenAI/whisper-tiny-encoder` / `-decoder` v1) transcribe each
turn; the UI paints a scrolling speaker-colored transcript, a who-spoke-when
timeline, and an "On-device · No cloud" badge + per-stage latency/RTF HUD. Fusion
is diarize-then-transcribe. Floor input is a bundled ≤10 s, 2-speaker, 16 kHz
mono clip. Deliverable: a screen-recordable demo video.

## Todo List

[x] Scaffold Flutter app (org `ai.zetic`, project `voxscribe`), PyroGuard/sibling
    structural template (loading screen, worker isolate, HUD, release iOS config).
[x] Pin `zetic_mlange: 1.8.1` exactly; assets wired (demo clip, vocab, mel filters).
[x] Pure-Dart log-mel (STFT n_fft=400/hop=160/Hann, 80-mel Slaney, log10,
    clamp max-8, (x+4)/4) — matches an offline golden reference vector.
[x] Pure-Dart GPT-2 byte-level BPE detokenizer (bundled `vocab.json`; skips
    specials id≥50257 / SOT 50258 / pad 50256).
[x] Pure-Dart preprocessing: WAV decode, stereo→mono ch0, linear resample to
    16 kHz, int16→/32768, 10 s window (160000), 30 s span pad (480000).
[x] Pure-Dart segmentation post-proc: 7-class powerset decode, frame→time map
    (589 frames), onset/offset state machine (min_on 0.30 s, min_off 0.50 s).
[x] Pure-Dart greedy 448-step decode (SOT seed, idx-1 logit row, EOT stop,
    int32 buffers) + repetition guard (30 s-pad hallucination mitigation).
[x] Diarize-then-transcribe fusion (attribution by construction).
[x] One long-lived worker isolate owns all 3 Melange handles (RUN_AUTO),
    warms each with a dummy inference, streams progress / lines / timings.
[x] Tier A: `flutter analyze` 0 issues; 14 trap tests (28 cases) green;
    iOS release build compiles (Runner.app, no-codesign); A4 benchmark recorded.
[x] Tier B: silence-frame DFT-skip optimization — A4 217 ms → 19 ms (exact).
[x] iOS signing scaffold (team WVJ22PPYBP, bundle `ai.zetic.voxscribe`, iOS 16.6);
    `--dart-define=MLANGE_KEY` injection; key never committed.
[ ] Physical-device run on iPhone (release, signed) — HUMAN, GATE 3.
[ ] [BLOCKED – human/dashboard] Confirm `ajayshah/PyannoteSegmentation` reached
    READY (was READY-pending at GATE 0; see R2 below) before the device run.
[ ] [BLOCKED – device] Confirm served `runtimeApType` per model on the console
    (benchmarked ≠ served); budget CPU-speed until NPU confirmed.
[ ] Replace the TTS demo clip with a higher-fidelity recording if desired
    (current clip is macOS `say`; regeneration steps below).
[ ] Stretch (post-floor): sliding-window segmentation, live mic, full
    diarization (4th embedding model + clustering) — separate GATE-0 upload.

## Tier A — Autonomous gates (ALL GREEN)

- **A1 Static analysis:** `flutter analyze` → **No issues found** (0 errors, 0
  warnings, 0 info). No TODOs/stubs in shipped `lib/`.
- **A2 Build:** `flutter build ios --release --no-codesign
  --dart-define=MLANGE_KEY=…` → **Built build/ios/iphoneos/Runner.app (29.8 MB)**.
  The vendored xcframework linked and the Xcode release build completed. (Signed
  install is the human's device step — see Tier C.)
- **A3 Unit tests (14 traps / 28 cases):** **all pass.** Files in `Flutter/test/`:
  resample, downmix (ch0), waveform_norm (int16/32768), segmentation_window
  (160000), frame_time_map (589 / 0.016875·f+0.0309688), powerset_decode (7
  classes incl. overlaps 4/5/6), logsoftmax (exp-sum=1, argmax-invariant),
  onset_offset_segment (merge ≤0.50 / drop <0.30), log_mel (golden vector,
  3000 frames), whisper_span_pad (480000→3000), greedy_decode (idx-1 row, EOT,
  SOT seed, repetition guard), token_dtype (int32, 1792 bytes), detokenizer
  (Ġ→space, specials skipped), fusion_attribution (2 segs → 2 tagged lines).
- **A4 Hot-path micro-benchmark:** `test/benchmark/hot_path_benchmark.dart`,
  full pure-Dart hot path (log-mel 480000→[1,80,3000] + powerset + onset/offset
  + greedy argmax over 51865×50 + detok), 9 iterations.
  **Median = 19.1 ms** (p90 21.3 ms). This is the Tier B baseline — the
  post-processing budget per span, NOT end-to-end device latency (native run()
  excluded; device-only).

## Tier B — Optimization log (0.5 % rule; A4 budget ≈ 19 ms; 0.5 % ≈ 0.1 ms)

| Lever | Action | Before→After (A4 median) | Verdict |
|---|---|---|---|
| **Log-mel silence-frame skip** | The 30 s span pad makes most STFT frames pure silence; an all-zero windowed segment has power 0 → mel 1e-10 → log10 = −10 exactly, so the 201-bin DFT is skipped for zero frames and the bins set to −10. Mathematically exact (golden test unchanged). | **217.3 ms → 19.1 ms (−198 ms, −91 %)** | **KEPT** (dominant win, ≫0.5 %) |
| **enc_hidden Tensor reuse** | The `[1,1500,384]` (576k floats) encoder hidden-states Tensor is built ONCE per span and reused across all 448 decode steps (SDK copies inputs into its own buffer each run). | Not on the A4 path (A4 mocks run()); removes 448× re-wrap of 576k floats on the real device decode loop. | **KEPT** (justified; device-side) |
| **Long-lived worker isolate** | One isolate created at app start owns all 3 handles for the app lifetime; per-call `compute()` isolate spawning is avoided (and is impossible — the SDK binds a handle to its creating isolate). Heavy tensors never cross the boundary; only segments/lines/timings do. | Structural; not A4-measurable. | **KEPT** (correctness + copy-cost) |
| **Pre-allocated typed buffers** | log-mel window/twiddle tables and the segment/log scratch buffers are allocated once; decode ids/mask are int32 buffers reused per step. Bulk typed-data ops, no boxed lists in hot loops. | Folded into the 19 ms figure. | **KEPT** |
| **Model warm-up** | One dummy inference per model right after load so the first real span is not the compile-on-first-run frame. | Device-only. | **KEPT** |
| **Direct DFT → real FFT** | A mixed-radix/Bluestein FFT for n_fft=400 could cut the per-(non-silent)-frame cost further. | Not implemented. | **SKIPPED** — after the silence-skip, log-mel touches only the ~200 non-silent frames of a span; remaining A4 budget is ~19 ms total and dominated by the greedy argmax over 51865×50, not the DFT. Adds significant complexity/bug-surface for sub-ms gain on the floor. Revisit only if longer spans (stretch) make it matter. |

## Tier C — Runtime-risk checklist (surfaced, NOT tested; the honest 70 %)

- **30 s silence-pad Whisper hallucination/looping (NAMED WATCH-ITEM).** Every
  speaker span — even a 1–2 s one — is zero-padded to 480000 samples (30 s)
  because the encoder input is fixed `[1,80,3000]`. Whisper can **hallucinate or
  loop tokens in the trailing silence**. **What the human watches for on device:**
  repeated/garbage tokens or a phrase repeating after the real utterance ends,
  especially on short spans. **Worker-side mitigation already in place:** (1) EOT
  termination (the loop stops on token 50257), and (2) a **repetition guard** —
  if the same token is emitted ≥24× in a row the decode stops early
  (`greedyDecode(repetitionGuard: 24)`). **Escalate** if hallucination persists
  beyond these: options are a `<|nospeech|>`/`avg_logprob` gate or trimming the
  pad to the span length server-side — both beyond the floor.
- **Served artifact (cannot be forced from the client).** Expected per model:
  segmentation has an LSTM that likely serves **CPU** (acceptable for one 10 s
  window, R3); Whisper encoder/decoder were READY at GATE 0 (85 % / 81 %, AUTO
  lands on NPU on capable silicon). Read the **actual** served `target`+`apType`
  (`runtimeApType=…`) from the native console — that, not the dashboard headline,
  is truth. Benchmarked ≠ served; budget CPU-speed until NPU is confirmed.
  Known crash path: FP32-GPU CoreML in MPSGraph on iOS/macOS 26.3+ — confirm no
  served artifact is GPU on affected OS; if it crashes, **escalate to ZETIC to
  filter GPU for that OS** (not client-fixable; no modelMode avoids it).
- **modelMode.** `RUN_AUTO` for all three (GATE-2 decision 9). modelMode does not
  steer off a crashing artifact.
- **Native observability.** Dart `print`/`debugPrint` does NOT surface on a
  release device console — only native logs do. On-device diagnostics
  (per-stage ms, RTF, segment count) are on the **HUD**, not logged. Watch the
  console during the run:
  ```
  xcrun devicectl device process launch --console --terminate-existing \
    --device <UDID> ai.zetic.voxscribe
  ```
  (`xcrun devicectl list devices` to get `<UDID>`.)
- **Signing / OS gates (manual, non-scriptable).** Team **WVJ22PPYBP**, bundle
  **ai.zetic.voxscribe**, `CODE_SIGN_STYLE=Automatic`; Developer Mode ON;
  "Trust" the developer cert; iOS **16.6+** (Podfile + pbxproj set to 16.6).
- **Build config.** Use **release** on device (debug hangs on launch on recent
  iOS/Xcode; a debug build's icon shows the "launch from Flutter tooling" screen
  — expected). Simulator is a dead end (device-only xcframework slice).
- **Key injection.** Add `--dart-define=MLANGE_KEY=<your_zetic_key>` to the
  build/run; the key is embedded in the client and is never committed (default
  sentinel `YOUR_MLANGE_KEY` makes a forgotten define fail loudly).
- **Network / cold start (3 model loads).** First launch downloads 3 models
  (2 reused Whisper + 1 segmentation) over the network → a spinner on poor
  Wi-Fi. Pre-download / pre-warm before the demo; warm-up runs one dummy
  inference per model after load.
- **R2 — segmentation READY.** `ajayshah/PyannoteSegmentation` was READY-pending
  (optimizing) at GATE 0. **Confirm it reached READY** before the device run, or
  model create() will fail/stall.
- **Non-determinism acceptance.** Server-side selection can return a different
  artifact minute to minute. Acceptance: runs cleanly across **multiple cold
  starts and at least one fresh install** before it counts as demo-ready;
  re-verify after any backend/model re-target.

## Exact human device-run steps

```bash
export PATH="/Users/ajayshah/development/flutter/bin:$PATH"
cd /Users/ajayshah/Desktop/ZETIC/voxscribe-wt/apps/VoxScribe/Flutter

# 1) Build & install (signed) — release, on a physical iPhone:
flutter run --release --dart-define=MLANGE_KEY=<your_zetic_key>
#    (or: flutter build ios --release --dart-define=MLANGE_KEY=<key>, then Xcode → Run)

# 2) Watch the native console for the served artifact + any crash:
xcrun devicectl list devices                     # get <UDID>
xcrun devicectl device process launch --console --terminate-existing \
  --device <UDID> ai.zetic.voxscribe

# 3) On screen: the demo clip auto-runs; transcript + timeline + HUD appear.
#    Confirm: 2 speakers labeled, overlap region near the end produces a
#    powerset overlap class, RTF/per-stage HUD populated, no token looping
#    in trailing silence. Tap ↻ to re-run. Record the screen.
```

## Regenerating the demo clip (if replacing the TTS clip)

The bundled `assets/demo_2spk.wav` was synthesized with macOS `say` (Samantha =
Speaker A, Daniel = Speaker B), assembled with the 4th line overlapping the tail
of the 3rd (so powerset overlap classes 4/5/6 fire), 16 kHz mono, ~8.3 s. To
regenerate or swap in a real recording, produce a ≤10 s, 16 kHz mono WAV with two
distinct voices and the structure ONE speaker → TWO → OVERLAP, and overwrite
`assets/demo_2spk.wav`. The script used: `say -v <Voice> -r 195 -o line.wav
--data-format=LEI16@16000 --file-format=WAVE "<text>"` per line, then mix/concat
with the 4th line starting ~0.35 s before the 3rd ends. Approved script:
- A: "Can you find a charging station near the airport?"
- B: "Sure, there's one about two miles ahead on the right."
- A: "Let's stop there before the—"
- B (overlapping A's tail): "—already adding it to the route."

## Deliverables

- Flutter source under `apps/VoxScribe/Flutter/` (main, loading/main screens,
  MelangeService (3 handles), pipeline_isolate, log_mel, detokenizer,
  preprocessor, postprocessor, diarization_fusion, models, transcript/timeline/
  HUD widgets).
- Assets: `assets/demo_2spk.wav`, `assets/vocab.json`, `assets/mel_filters_80.bin`.
- Asset/golden generators: `Flutter/tool/gen_mel_filters.py`,
  `Flutter/tool/gen_logmel_golden.py`.
- Tier A test suite (14 traps) + A4 micro-benchmark in `Flutter/test/`.
- iOS config: team WVJ22PPYBP, bundle ai.zetic.voxscribe, Podfile/pbxproj iOS 16.6.
- `BUILD_PLAN.md` (GATE-2), this `HANDOFF.md`, `Flutter/README.md`.

## References

- App directory: `apps/VoxScribe` (branch `app/voxscribe`).
- Spec: `apps/VoxScribe/SPEC.md`; plan: `apps/VoxScribe/BUILD_PLAN.md`.
- SDK: ZETIC Melange (`zetic_mlange 1.8.1`, Flutter FFI). `ZeticMLangeModel.create`
  (async) / `model.run(List<Tensor>)` (sync) / `Tensor.float32List` /
  `Tensor.int32List` / `model.close()`.
- Models (all v1): `ajayshah/PyannoteSegmentation` (x[1,1,160000] → [1,589,7]
  log-softmax); `OpenAI/whisper-tiny-encoder` (input_features[1,80,3000] →
  [1,1500,384]); `OpenAI/whisper-tiny-decoder` (ids int32[1,448],
  enc_hidden[1,1500,384], enc_mask int32[1,448] → [1,448,51865]).
- Native reference ported to Dart: `apps/whisper-tiny` (WhisperFeature/Decoder,
  vocab.json); structural template: sibling YOLO apps (PyroGuard-derived).
- Test device: TBD (iPhone, iOS 16.6+).
```
