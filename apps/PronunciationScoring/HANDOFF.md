Goal
A fully on-device pronunciation-coach demo for Flutter (iOS-first): the user
reads a displayed sentence into the mic during a fixed 5.11 s recording
window; a 40 MB Citrinet-256 ARPABET phoneme-CTC model (via ZETIC Melange,
ajayshah/PronunciationScoring v1, RUN_AUTO) produces phoneme log-posteriors;
a pure-Dart scoring head (CTC Viterbi forced alignment + per-phoneme GOP)
renders per-word / per-phoneme color-coded scores, an overall score, and a
latency/served-artifact HUD. Product name: "SayRight" (display name only;
folder, bundle id, and Melange model name unchanged).

Orchestrator ruling on record (GATE 0): STAY WITH CITRINET (40 MB, measured
PER 18.5%). HuBERT-base-phoneme (377 MB, PER 11.4%) fails the mobile-size
rubric and is the escalation path ONLY if device results disappoint. The demo
story is GOP scoring (measured 5.3-13x correct-vs-mismatch separation), not
raw transcription.

Todo List
[x] Stage 0: model discovery, head-to-head bakeoff (Citrinet 18.5% PER vs
    HuBERT 11.4%), winner selection (model_selection.md).
[x] Stage 0: export citrinet256_phoneme.onnx — raw-waveform float32[1,81760]
    -> logprobs float32[1,64,45], opset 12, static shapes, NeMo log-mel
    frontend baked into the graph, parity vs NeMo 9.2e-4 (export.py).
[x] Stage 0: behavioral validation on real speech via onnxruntime — greedy
    decode tracks CMUdict targets; GOP separation 5.3-13x; zero-padding trap
    found and measured (validation/validate_onnx.py + reference wavs).
[x] GATE 0: registered ajayshah/PronunciationScoring v1, READY; served shapes
    echo the export; RUN_AUTO; Apple AUTO latency expectation ~50-70 ms per
    inference (acceptable: one inference per recording).
[x] SPEC.md finalized (all GATE-0 blanks filled, no TBDs).
[ ] GATE 2: build plan + Tier A test list + this initial ticket presented;
    awaiting orchestrator approval. NO APP CODE before approval.
[ ] Flutter scaffold (apps/PronunciationScoring/Flutter/): loading screen
    (model download + warm-up), main screen (sentence card, record ring,
    results view), theme, HUD.
[ ] Secrets hygiene: gitignored lib/config/secrets.dart (personal key) +
    committed secrets.example.dart placeholder; git check-ignore verified
    before every commit.
[ ] Demo sentence asset: ~8 curated sentences sized to read in 3.5-5 s, with
    precomputed ARPABET id sequences (CMUdict, stress stripped) generated
    offline by a committed tools script.
[ ] Mic capture service: 16 kHz mono PCM16 stream, records the FULL 5.11 s
    window (exactly 81760 samples), asserts actual sample rate, never
    zero-pads (noise-pad ~1e-3 RMS only for OS-truncated captures).
[ ] Melange lifecycle wrapper: create(personalKey:, name:
    'ajayshah/PronunciationScoring') -> warm-up dummy inference ->
    Tensor.float32List [1,81760] -> run -> asFloat32List -> close; verify
    installed zetic_mlange API surface before coding against it.
[ ] Preprocessor: PCM16 -> float32 /32768.0, exact 81760-sample contract.
[ ] Scoring head (pure Dart, contract = validation/validate_onnx.py):
    frame-major [64][45] view, CTC Viterbi forced alignment (blank=44,
    skip-rule per spec), per-phoneme GOP = mean aligned posterior, word =
    mean of phones (min-phone highlight), overall = fill-aware calibrated
    mapping using blank-frame fraction as window-fill proxy.
[ ] Optional greedy "what we heard" decode behind a details expander
    (decoration only; scoring never uses it).
[ ] Golden fixtures: run validation harness on reference wavs, export JSON
    (aligned frame sets + GOP scores + greedy strings); Dart tests reproduce
    within 1e-3.
[ ] Tier A battery green: flutter analyze clean, release device build, unit
    tests (list at GATE 2), hot-path micro-benchmark recorded.
[ ] Tier B optimization pass with before/after micro-benchmark deltas (or
    justified skips — note: one inference per recording, not per-frame, so
    the hot path is the scoring head + buffer handling).
[ ] Custom launcher icon (flutter_launcher_icons, 1024x1024 source,
    remove_alpha_ios: true) + "SayRight" display name (CFBundleDisplayName,
    android:label, in-app title).
[ ] iOS config: signing, NSMicrophoneUsageDescription, iOS 16.6 min, release
    build on physical iPhone; device-console capture workflow documented.
[ ] Tier C runtime-risk checklist filled for GATE 3.
[ ] [BLOCKED – human] GATE 3 physical-device run (mic + Melange serving only
    observable on hardware; iOS simulator is a dead end — device-only
    xcframework slice).
[ ] [BLOCKED – ZETIC backend, accepted] Apple RUN_AUTO serves CPU-class
    (~52-77 ms benched) while NPU-class (~6.6-14 ms) exists under SPEED.
    Accepted for this app (single inference per recording); revisit with
    ZETIC only if device UX suffers.

Deliverables
- Flutter source under apps/PronunciationScoring/Flutter/ (screens, services:
  melange/audio/preprocess/aligner/scorer, models, widgets, tests incl.
  golden fixtures and hot-path benchmark).
- Stage-0 artifacts (committed): export.py, labels.txt, melange_upload.md,
  model_selection.md, SPEC.md, validation/ harness + reference wavs.
  (citrinet256_phoneme.onnx + sample_input.npy on worktree disk; *.onnx/*.npy
  are repo-gitignored by policy — regenerable via export.py.)
- Registered Melange model: ajayshah/PronunciationScoring v1 (READY).
- This HANDOFF.md kept live through the build; finalized at GATE 3.

References
- App directory: apps/PronunciationScoring
- Core SDK: ZETIC Melange (zetic_mlange Flutter plugin; verify installed
  version's API before coding)
- Model: Peacockery/citrinet-256-phoneme-en (MIT) — Citrinet-256 ARPABET-41
  phoneme CTC; input float32[1,81760] raw 16 kHz mono waveform; output
  float32[1,64,45] log-softmax; labels.txt authoritative (id0=AA, blank=44)
- Scoring contract: validation/validate_onnx.py (CTC forced alignment + GOP)
- Doc set: apps/agentic-workflow-docs/ (CLAUDE.md, AGENTS.md, VALIDATION.md,
  EXPLORATION.md)
- Test device: human's iPhone (per PyroGuard: iPhone 15, iOS 26.x — confirm
  at GATE 3)
