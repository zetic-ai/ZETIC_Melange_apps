# AGENTS.md — Orchestration Protocol

Delegation here is manual and deliberate. The orchestrator decides when to hand off, what to hand off, and when to resume. Nothing spawns automatically.

---

## Agent roster

| Agent | Model / effort | Owns | Writes app code? |
|---|---|---|---|
| Master orchestrator | Opus 4.8, maximum reasoning effort | Specs, delegation, gates, review | No |
| Explorer (one per use-case) | Opus 4.8, high reasoning effort | Stage 0: find + export one model, one folder | No |
| Worker (one per app) | Opus 4.8, high reasoning effort | One app, one branch | Yes |

Explorers and workers are separate roles. A run's Explorers all target the **same
technology family** (see EXPLORATION.md). The orchestrator may let an Explorer's
session continue as that app's worker after GATE 0, or spawn a fresh worker — either
is fine, since each app has its own branch/worktree.

Set the model and reasoning effort per the Claude Code workflows docs (code.claude.com/docs/en/workflows). Confirm the exact invocation there; the intent is "max-effort planner, high-effort builders." Do not assume flag names; verify them.

---

## The orchestrator's loop

1. **Stage 0 (exploration).** Pick one technology family and the target use-cases for this run. Spin off one Explorer per use-case (see EXPLORATION.md). Each searches Hugging Face, picks the best model for Melange, exports it, and populates its app folder with the ONNX, `sample_input.npy`, `melange_upload.md`, `model_selection.md`, and a pre-drafted spec stub. All Explorers in a run share one technology family (one export recipe).
2. **GATE 0 (Melange upload):** each Explorer stops and hands the human its `melange_upload.md`. The human drags the two artifacts into the dashboard, registers the model, waits for READY, and pastes back the registered model name + version and the served input/output shapes. The app is blocked until then.
3. Finalize the per-app spec: merge the Explorer's stub with the human's GATE-0 paste-back so every section of the CLAUDE.md section 6 template is filled. A gap here becomes a guess in a dark worker session.
4. **GATE 1 (spec approval):** present the finalized spec to the human. Do not delegate to a worker until approved.
5. Hand the approved spec to a worker on a fresh branch. The worker plans and proposes its test list.
6. **GATE 2 (approach approval):** the worker returns a build plan plus the Tier A test list it intends to write, before writing app code. Human approves or redirects.
7. Worker runs dark: writes the app, writes the tests, runs the validation loop until the Tier A battery passes and the Tier B optimization checklist is satisfied.
8. **GATE 3 (handoff for device run):** the worker returns the validation report plus the Tier C runtime-risk checklist. The human performs the physical-device run. The worker never claims "done"; it claims "ready for device."
9. Human reports device results. If a device-only issue appears, the orchestrator decides whether it is a worker fix (Dart pipeline) or a human/dashboard action (artifact retarget, OS trap).

Multiple Explorers and workers may be at different gates at the same time. The orchestrator tracks each app's gate state (GATE 0 through GATE 3).

---

## Branch / worktree mechanics

Each app gets its own branch and its own working directory so parallel workers never collide.

```bash
# from the repo root, one per app
git worktree add ../pyroguard-wt   -b app/pyroguard
git worktree add ../audiotag-wt    -b app/audiotagger
# ...one worktree per app
```

Launch one worker session per worktree directory. Each worker:
- works only inside its own worktree,
- commits to its own `app/<name>` branch,
- never touches another app's files.

Merge to main only after a successful human device run, not at GATE 3.

---

## Handoff contract (what an agent returns at each gate)

**At GATE 0 (Explorer):**
- The populated app folder: `export.py`, `<model>.onnx`, `sample_input.npy`.
- `melange_upload.md` — the exact dashboard steps plus the fields the human must paste back (model name/version, served shapes, modelMode default RUN_AUTO).
- `model_selection.md` — the top-5 shortlist, scoring, and winner rationale.
- A pre-drafted SPEC stub with the GATE-0 fields (Melange name/version, served shapes) left blank.
- The Explorer presents these and stops. The human does the dashboard upload; see EXPLORATION.md.

**At GATE 2:**
- A short build plan (files, pipeline approach, threading model).
- The exact Tier A test list it will write.
- Any spec ambiguity it found (this is the worker's one chance to ask before going dark).

**At GATE 3:**
- Validation report: Tier A results (analyze, build, unit tests, micro-benchmark numbers).
- Tier B optimization log: each optimization applied, with its measured delta on the Dart hot-path micro-benchmark, or a justification for skipping.
- Tier C runtime-risk checklist (from VALIDATION.md), filled for this app: served-artifact expectation, modelMode chosen, device-console command to watch, signing/build-config notes, network/cold-start risk, and the "run it N times" acceptance note.
- A `HANDOFF.md` written in the Jira ticket format below.

The worker presents these and stops. The human runs the device.

---

## Handoff ticket format (HANDOFF.md)

At GATE 3 the worker writes a `HANDOFF.md` in the project folder using the Jira structure below. This keeps every app's handoff paste-ready into the real tracker, and forces the worker to state plainly what is done, what is blocked, and what the human must do next. The Todo list uses `[x]` for completed, `[ ]` for open, and `[ ] [BLOCKED – owner]` for anything the worker cannot resolve from the app side (for example a server-side artifact issue). Blocked items must name the root cause and the owner.

The sections are: Goal, Todo List, Deliverables, References. Include the test device when known.

### Worked example (PyroGuard HANDOFF.md)

```
Goal
A real-time, fully on-device fire & smoke detection demo for Flutter (iOS),
powered by a YOLO11s detector through the ZETIC Melange SDK. Streams the live
camera feed, runs detection each frame on-device, and overlays
labeled fire/smoke boxes with a live latency + detection-count HUD.

Todo List
[x] Create core Flutter structure (loading screen, camera screen, theme, HUD).
[x] Export YOLO11s (leeyunjai/yolo11-firedetect) to ONNX (opset 12, 640x640)
    and register on Melange (ajayshah/FireDetectionYOLO, v1).
[x] Melange lifecycle wrapper (create -> Tensor.float32List -> run -> close).
[x] Preprocessing: letterbox 640x640, BGRA (iOS) / YUV420 (Android) decode,
    NCHW float32 normalization.
[x] Post-processing: decode [1,6,8400] channel-major, threshold, un-letterbox,
    per-class NMS.
[x] Detection overlay (rotation + BoxFit.cover mapping) and HUD.
[x] iOS signing/deploy (team, NSCameraUsageDescription, iOS 16.6 min); run on
    a physical iPhone.
[x] Resolve device-only xcframework (no simulator slice) via physical-device +
    release-mode builds.
[x] Per-stage latency profiler (preprocess / run / postprocess) and
    device-console crash capture.
[x] Inference-time crash (RESOLVED by ZETIC): Melange was serving a
    COREML_FP32/GPU artifact that aborted in Apple MPSGraph on iOS 26.3+
    (MLIR pass manager failed, SIGABRT). No client modelMode avoided it — all
    four (AUTO/ACCURACY/SPEED/QUANTIZED) resolved to the same artifact. ZETIC
    filtered the GPU candidate server-side for affected OS versions; now serves
    TFLITE_FP16/CPU, no crash.
[x] Fix accuracy: the camera buffer is already delivered upright (720x1280), so
    removed the SPURIOUS 90-degree overlay rotation that transposed every box
    into a tall sliver. No input rotation was needed.
[ ] [BLOCKED – ZETIC backend] Inference latency (~400ms) is CPU-bound: the
    served TFLITE_FP16/CPU artifact benchmarks ~383ms; the Dart pipeline is only
    ~20ms. Real fix is a CoreML / Neural-Engine artifact from ZETIC (~3ms);
    runAuto will auto-select it once served. Minor secondary (worker-side):
    replace the per-frame compute() double-isolate spawn (~20ms).
[ ] Android run verification once iOS is stable.
[ ] Static sample-image validation harness for pre/post-processing.

Deliverables
- Flutter source under FireDetectionYOLO/Flutter/ (screens, MelangeService,
  preprocessor, postprocessor, NMS, detection model, overlay/HUD).
- Model assets: export.py, firedetect-11s.onnx, sample_input.npy, registered
  Melange model (ajayshah/FireDetectionYOLO v1).
- iOS config: signing (team WVJ22PPYBP), Info.plist camera usage, Podfile
  (iOS 16.6, vendored ZeticMLange.xcframework).
- Diagnostics: HANDOFF.md (root-cause analysis, backend-selection test matrix,
  resume checklist), in-code latency profiler, devicectl console workflow.

References
- App directory: apps/FireDetectionYOLO
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI)
- Model: YOLO11s fire/smoke — leeyunjai/yolo11-firedetect
  (input float32[1,3,640,640], output float32[1,6,8400], classes: fire, smoke)
- Frameworks: Flutter, camera plugin, CoreML / Apple Neural Engine (via
  Melange), Ultralytics (export)
- Test device: iPhone 15 (iPhone15,4, A16), iOS 26.5
```