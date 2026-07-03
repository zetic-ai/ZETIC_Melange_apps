## Goal

A fully on-device pronunciation-coach demo for Flutter (iOS-first): the user reads a displayed sentence into the mic during a fixed 5.11 s recording window; a 40 MB Citrinet-256 ARPABET phoneme-CTC model (via ZETIC Melange, ajayshah/PronunciationScoring v1, RUN_AUTO) produces phoneme log-posteriors; a pure-Dart scoring head (CTC Viterbi forced alignment + per-phoneme GOP) renders per-word / per-phoneme color-coded scores, an overall score, and a latency/served-artifact HUD. Product name: "SayRight" (display name only; folder, bundle id, and Melange model name unchanged).

Orchestrator ruling on record (GATE 0): STAY WITH CITRINET (40 MB, measured PER 18.5%). HuBERT-base-phoneme (377 MB, PER 11.4%) fails the mobile-size rubric and is the escalation path ONLY if device results disappoint. The demo story is GOP scoring (measured 5.3-13x correct-vs-mismatch separation), not raw transcription.

## Todo List

- [x] Stage 0: model discovery, head-to-head bakeoff (Citrinet 18.5% PER vs HuBERT 11.4%), winner selection (model_selection.md).
- [x] Stage 0: export citrinet256_phoneme.onnx — raw-waveform float32[1,81760] -> logprobs float32[1,64,45], opset 12, static shapes, NeMo log-mel frontend baked into the graph, parity vs NeMo 9.2e-4 (export.py).
- [x] Stage 0: behavioral validation on real speech via onnxruntime — greedy decode tracks CMUdict targets; GOP separation 5.3-13x; zero-padding trap found and measured (validation/validate_onnx.py + reference wavs).
- [x] GATE 0: registered ajayshah/PronunciationScoring v1, READY; served shapes echo the export; RUN_AUTO; Apple AUTO latency expectation ~50-70 ms per inference (acceptable: one inference per recording).
- [x] SPEC.md finalized (all GATE-0 blanks filled, no TBDs).
- [x] GATE 2: build plan + Tier A test list approved; app built exactly to plan.
- [x] Flutter scaffold (apps/PronunciationScoring/Flutter/): loading screen (model download + warm-up), main screen (sentence card, record ring, results view), theme, HUD. (org com.zeticai, package sayright.)
- [x] Secrets hygiene: gitignored lib/config/secrets.dart (personal key) + committed secrets.example.dart placeholder; .gitignore rule added, git check-ignore verified; key absent from all tracked files (grep clean).
- [x] Demo sentence asset: 8 curated sentences (incl. on-device-AI / industrial flavor), each 37-44 phones -> est 3.88-4.62 s read time, with precomputed ARPABET id sequences + per-word spans generated offline by tools/gen_sentences.py (CMUdict, stress stripped, first pronunciation).
- [x] Mic capture service (audio_recorder.dart): mono PCM16 stream, records the FULL 5.11 s window; cancel discards (never scores a partial). Requests 16 kHz native; 48 kHz->16 kHz decimation path wired + HUD-visible.
- [x] Melange lifecycle wrapper (melange_service.dart): create(personalKey:, name:'ajayshah/PronunciationScoring', version:1, modelMode: runAuto) -> warm-up dummy inference -> Tensor.float32List [1,81760] -> run -> asFloat32List (copied out of reused buffer) -> score. API surface verified against installed zetic_mlange 1.8.1.
- [x] Preprocessor (preprocessor.dart): PCM16 -> float32 /32768.0, exact 81760 contract, noise-pad tail (never zero-run), proper 31-tap FIR decimation, sample-rate refusal (only 16k/48k).
- [x] Scoring head (pure Dart, contract = validation/validate_onnx.py): frame-major [64][45] view (postprocessor.dart), CTC Viterbi forced alignment (ctc_aligner.dart, blank=44, skip-rule per spec), per-phoneme GOP = mean exp(logprob) over aligned frames, word = mean of phones (min-phone highlight), overall = fill-aware calibrated mapping using blank-frame fraction as window-fill proxy (gop_scorer.dart).
- [x] Greedy "what we heard" decode behind a details expander in score_view.dart (decoration only; scoring uses aligned GOP).
- [x] Golden fixtures: validation/export_golden.py runs the ONNX on the reference wavs -> test/fixtures/golden_{ls1,ref1,ref2}.json (logprobs + aligned frames + GOP + greedy + blank fraction); golden_parity_test reproduces greedy/alignment exactly and GOP within 1e-3.
- [x] Tier A battery green: flutter analyze clean (0/0/0); 88 unit tests pass; hot-path micro-benchmark recorded (median 28us). iOS release build: see below.
- [x] Tier B optimization pass — see "Tier B log" below.
- [x] Custom launcher icon (flutter_launcher_icons, 1024x1024 source, remove_alpha_ios: true) + "SayRight" display name (CFBundleDisplayName, android:label, in-app title).
- [x] iOS config: NSMicrophoneUsageDescription, iOS 16.6 min (Podfile + pbxproj), debug-signed release; release build --no-codesign (A2 gate).
- [x] Tier C runtime-risk checklist filled for GATE 3 — see below.
- [ ] **[BLOCKED – human]** GATE 3 physical-device run (mic + Melange serving only observable on hardware; iOS simulator is a dead end — device-only xcframework slice).
- [ ] **[BLOCKED – ZETIC backend, accepted]** Apple RUN_AUTO serves CPU-class (~52-77 ms benched) while NPU-class (~6.6-14 ms) exists under SPEED. Accepted for this app (single inference per recording); revisit with ZETIC only if device UX suffers.

## Deliverables

- Flutter source under apps/PronunciationScoring/Flutter/ (screens, services: melange/audio/preprocess/aligner/scorer, models, widgets, tests incl. golden fixtures and hot-path benchmark).
- Stage-0 artifacts (committed): export.py, labels.txt, melange_upload.md, model_selection.md, SPEC.md, validation/ harness + reference wavs. (citrinet256_phoneme.onnx + sample_input.npy on worktree disk; *.onnx/*.npy are repo-gitignored by policy — regenerable via export.py.)
- Registered Melange model: ajayshah/PronunciationScoring v1 (READY).
- This HANDOFF.md kept live through the build; finalized at GATE 3.

## GATE 3 validation results (A1–A4)

- A1 analyze: `flutter analyze` -> "No issues found!" (0 errors / 0 warnings / 0 infos).
- A2 iOS release build: `flutter build ios --release --no-codesign` -> Built build/ios/iphoneos/Runner.app (28.9 MB). Bundle id com.zeticai.sayright, CFBundleDisplayName "SayRight", NSMicrophoneUsageDescription present, iOS 16.6 min, custom launcher icon applied. (Must be codesigned before device deploy.)
- A3 tests: `flutter test` -> All 88 tests pass. Full Tier A battery green: labels, tensor_layout, logprob_semantics, greedy_collapse, ctc_alignment_hand, ctc_skip_rule, golden_parity (ls1/ref1/ref2 within 1e-3), preprocessor_contract, decimation, window_fill_proxy, score_aggregation, sentence_asset.
- A4 micro-benchmark: hot path (greedy decode + CTC forced alignment + GOP over a ~40-phone target on a mock [64,45]) median 28 us, p95 36 us over 2000 iters (test/benchmark/hot_path_benchmark.dart).

## Tier B log (optimization pass, with deltas)

Context: exactly ONE model inference per 5.11 s recording; the Dart hot path is the scoring head + buffer handling, already 28 us median. Checklist applied:

- Threading: scoring runs on the MAIN isolate by design. A per-run compute() isolate would add ~hundreds of us of spawn + copy overhead to a 28 us job — net LOSS. Justified skip (measured hot path << isolate hand-off cost).
- Buffer copies: model output is copied out of the reused native buffer exactly once (Float32List.fromList) before scoring — required for correctness, not removable. Input tensor built once per recording.
- Pre-allocation: aligner DP grids are the only per-run allocation (64 x (2N+1) doubles, N<=~48). At 28 us median this is not a bottleneck; pre-allocating across runs would micro-optimize a non-hot path — deferred, not justified to complicate.
- FIR decimation taps: computed ONCE (cached top-level) and reused per capture, not rebuilt per call. Applied.
- No per-frame work on the UI thread; results are immutable value types.

Net: no code change delivered more than noise against the 28 us baseline; the optimization budget is dominated by the ~50-70 ms served inference, which is a ZETIC backend concern (see accepted BLOCKED item).

## Tier C runtime-risk checklist (device-only, for GATE 3 human run)

- Device console: mic + Melange serving are observable only on hardware (iOS simulator is a dead end — device-only xcframework slice). After codesigning & installing, capture the native console with: `xcrun devicectl device console --device <UDID>` (# or Console.app, filter "Runner"). Read the SERVED artifact line ZETIC logs at model create.
- Served-artifact expectation: RUN_AUTO on Apple serves a CPU-class artifact, ~50-70 ms per inference (NPU-class ~6.6-14 ms exists only under SPEED). FINE — one inference per recording. Surface measured latency + served string on the HUD (already wired).
- Mic-permission gate: first record tap triggers the iOS permission prompt (NSMicrophoneUsageDescription). Denial path shows an in-app error, no crash.
- Cold start: first launch downloads the optimized model binary (~10-38 MB, network-dependent) via Melange; loading screen shows progress, then a warm-up dummy inference. Needs INTERNET (Android permission added) + connectivity.
- Selection non-determinism: the backend may serve a different artifact/mode across launches/OS versions; the HUD's served-artifact + latency readout is the single source of truth on device — do not assume a fixed backend.
- Memory: up to ~273 MB at load/inference (paste-back); no app-side action.

## References

- App directory: apps/PronunciationScoring
- Core SDK: ZETIC Melange (zetic_mlange Flutter plugin; verify installed version's API before coding)
- Model: Peacockery/citrinet-256-phoneme-en (MIT) — Citrinet-256 ARPABET-41 phoneme CTC; input float32[1,81760] raw 16 kHz mono waveform; output float32[1,64,45] log-softmax; labels.txt authoritative (id0=AA, blank=44)
- Scoring contract: validation/validate_onnx.py (CTC forced alignment + GOP)
- Doc set: apps/agentic-workflow-docs/ (CLAUDE.md, AGENTS.md, VALIDATION.md, EXPLORATION.md)
- Test device: human's iPhone (per PyroGuard: iPhone 15, iOS 26.x — confirm at GATE 3)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
