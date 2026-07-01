# CLAUDE.md — ZETIC On-Device ML Demo Apps

This file is shared context for every session in this repo. Read it fully before acting.

**The doc set:** CLAUDE.md (this file — shared context + SDK realities) · EXPLORATION.md (Stage 0: find + export a model) · AGENTS.md (orchestration protocol + gates) · VALIDATION.md (the quality battery). Read all four before running the workflow.

---

## 1. What this repo is

A collection of small, polished Flutter demo apps, each wrapping one on-device ML model deployed through ZETIC's Melange SDK. The apps exist to support ZETIC's go-to-market motion (VivaTech leads, industrial and automotive prospects). Each app should look trade-show ready, not prototype-grade.

Every app follows the same anatomy: one Melange model, a Flutter UI, and a pure-Dart pre/post-processing pipeline. The first reference implementation is PyroGuard (YOLO11s fire and smoke detector).

---

## 2. ZETIC / Melange in one paragraph

Melange automatically compiles a trained ONNX model into NPU-optimized binaries, benchmarks them across a real device farm, and serves each end-user device the best-performing binary at runtime. The developer uploads an ONNX plus a sample input on the dashboard, waits for the CONVERTING then OPTIMIZING then READY cycle, then pulls the model into the app with the SDK. NPU acceleration is the whole point: on flagship silicon, inference can drop from hundreds of CPU milliseconds to single-digit NPU milliseconds.

That is the ideal. In practice (see section 5) "best-performing" is not guaranteed: selection can serve a crashing artifact, or fall back to CPU rather than the NPU, and a benchmarked NPU row may never be served for a given chip. Treat the served artifact on the device console — not the dashboard's headline number — as ground truth, and budget for a CPU-speed fallback until the NPU path is confirmed on hardware.

---

## 3. Roles

**Human (Ajay).** Owns everything the agents cannot do: the **Melange dashboard upload and model creation** (GATE 0 — the one manual step in the pipeline), physical-device runs on the iPhone, and approving each checkpoint. The human is the only one who sees real on-device behavior. (ONNX export and per-app folder creation are now done by the Explorer in Stage 0; the human just drags the artifacts into the dashboard.)

**Master orchestrator (Opus 4.8, maximum reasoning effort).** Does not write app code. Its job is to run Stage 0 exploration (spin off Explorers, see EXPLORATION.md), turn each chosen model into a complete, gap-free per-app spec (template in section 6), then delegate to one worker per app, hold the checkpoint gates, and review what each agent returns. It delegates all task-specific work (sourcing, edits, builds, research, validation) to agents and does not execute it inline in the main loop — its own job is only to decide, delegate, review, and hold gates. Delegation is manual and deliberate, never automatic. See AGENTS.md.

**Explorer agent (Opus 4.8, high reasoning effort), one per use-case.** Stage 0 only. Searches Hugging Face for its assigned use-case, reasons which model is best for Melange, exports it to ONNX, generates the sample input, and populates the app folder (artifacts + `melange_upload.md` + `model_selection.md` + a pre-drafted spec stub). Does not write app code. Stops at GATE 0 and tells the human exactly what to upload and paste back. See EXPLORATION.md.

**Worker agent (Opus 4.8, high reasoning effort), one per app.** Owns a single app on its own git branch. Tries to one-shot the app from the spec, runs the full validation loop against VALIDATION.md on its own, and stops at the defined gates rather than dumping unverified work. A worker may run its validation loop as a subroutine (write, test, read failures, fix, repeat) but does not silently push past a gate.

---

## 4. Standard app anatomy

```
<AppName>/                     # the app folder (created by the Explorer in Stage 0)
  export.py                    # Stage 0: re-runnable export recipe
  <model>.onnx                 # Stage 0: exported artifact (drag into dashboard)
  sample_input.npy             # Stage 0: sample input (drag into dashboard)
  melange_upload.md            # Stage 0: the human's GATE-0 dashboard instructions
  model_selection.md           # Stage 0: top-5 shortlist + winner rationale
  HANDOFF.md                   # living Jira ticket — created first (after GATE 1), finalized at GATE 3
  Flutter/                     # the app itself (worker-owned, built after GATE 0)
    assets/
      icon/
        app_icon.png           # 1024x1024 source for the launcher icon
                               #   (generates iOS AppIcon.appiconset + Android mipmaps)
    lib/
      main.dart
      screens/
        loading_screen.dart    # model download + warm-up, progress bar
        main_screen.dart       # the live demo (camera, mic, etc.)
      services/
        melange_service.dart   # model init, run, warm-up, close
        preprocessor.dart      # raw input -> List<Tensor>
        postprocessor.dart     # raw output -> List<Detection/Event/etc.>
        <task_specific>.dart   # e.g. nms.dart for detectors
      models/
        <result_type>.dart     # data class for one result
      widgets/
        <overlay/feed/hud>.dart
    test/
      <unit tests per VALIDATION.md Tier A>
      benchmark/
        hot_path_benchmark.dart  # mock-tensor micro-benchmark
```

**Binding: every app ships a custom launcher icon.** Each app ships a custom launcher icon that visually identifies the app's domain (e.g. a shelf/planogram glyph, a license-plate glyph, a drone/aerial glyph), generated from a 1024x1024 source (`assets/icon/app_icon.png`) via the `flutter_launcher_icons` package for both iOS and Android (`remove_alpha_ios: true`). The default Flutter icon is not acceptable for a trade-show build.

**Binding: every app ships a cool, domain-identifying product name.** Each app gets a short, memorable product name distinct from its model/folder name (e.g. the diarization app is "VoxScribe"; YOLO examples: shelf→"ShelfSense", license-plate→"PlateHawk", aerial→"SkyScout"). The orchestrator (or its agent) just picks one — no need to confirm names with the human. Apply it as the user-facing display name only — set iOS `CFBundleDisplayName`, Android `android:label`, and the in-app title (`MaterialApp(title:)`, app-bar/loading text) — and do NOT change the bundle id, the app folder name, or the registered Melange model name (`ajayshah/<ModelName>`), which stay stable. The default `flutter create` project name (e.g. "Runner"/the package name) is not acceptable for a trade-show build.

---

## 5. Hard-won SDK and platform realities (binding constraints)

These were learned the painful way on PyroGuard. Treat them as non-negotiable facts, not suggestions.

**SDK API (zetic_mlange, verify the installed version before coding).**
- Model load is an async factory: `await ZeticMLangeModel.create(personalKey: ..., name: ...)`. The older `ZeticMLangeModel(...)` constructor with `version`, `modelMode`, and `onDownload` is not the shipping basic API. Confirm the exact surface of whatever version is installed before writing against it.
- Inference takes typed tensors: `model.run(List<Tensor>)`, build inputs with `Tensor.float32List(data, shape: ...)`, read outputs with `outputs.first.asFloat32List()`.
- Tear down with `model.close()`.
- The Flutter plugin bundles its own native deps. Do not hand-pin native versions; that causes conflicts.

**Backend selection is server-side and not steerable from the client.** The choice of NPU vs GPU vs CPU and the precision (FP32, FP16, INT8) is decided on ZETIC's servers. Client-side `target` / `apType` / `quantType` overrides are forwarded but currently ignored; only `modelMode` reaches the selector. You are tuning a system you cannot fully see.

**A served artifact can crash at inference even after loading cleanly.** On PyroGuard, an FP32-GPU CoreML artifact loaded fine ("BackendSelectionExecutor: success"), then aborted at the *first inference* inside Apple's MPSGraph compiler on iOS/macOS 26.3+ (`MLIR pass manager failed`, SIGABRT, uncatchable in Dart). It is an Apple GPU-compiler bug (a fusion-pattern bug hit by standard ViT/YOLO-style attention heads), not a bad model — the same graph runs fine on CPU and the Neural Engine. The SDK only falls back on a *load* failure, never on a runtime crash, so it cannot self-recover.
- **Client `modelMode` did NOT fix this.** All four modes (RUN_AUTO, RUN_ACCURACY, RUN_SPEED, RUN_QUANTIZED) were tested on-device, and every one returned the *same* crashing FP32-GPU artifact for this chip. Do not expect a modelMode to steer off a crashing artifact — it usually can't, because the server returns the same top-ranked candidate regardless.
- **The durable fix was server-side, by ZETIC:** they filter the GPU candidate out of backend selection for the affected OS versions. The selection request already includes `os_version`; the fix was a backend change on their side. If you hit a GPU/MPSGraph crash on a new OS, escalate to ZETIC to filter GPU for that OS — it is genuinely not fixable from the client.
- Record which modelMode you requested, but treat the **served artifact** (target + apType, read from the native console) as the source of truth — not the mode you asked for.

**Filtering GPU can drop you to CPU, not the NPU.** After ZETIC filtered GPU for PyroGuard, every modelMode fell back to `TFLITE_FP16 / CPU` (~383 ms benchmarked), *not* the Neural Engine. "Removed the crash" and "got NPU speed" are two separate wins — the second may require ZETIC to compile/serve a CoreML artifact targeting the Neural Engine (`ComputeUnit.all`/NPU). Plan for CPU-speed as the realistic default until the NE artifact is confirmed on the device console (`runtimeApType=NPU`).

**"Benchmarked" is not the same as "deployable."** The dashboard may show a fast row (for example NPU FP16 at single-digit ms) that the selection API never actually serves for a given chip. On PyroGuard the 2.86 ms NPU-FP16 row was real in the report but never served — every mode resolved to GPU (then, post-fix, CPU). Do not assume the headline number is what runs.

**The iOS simulator is a dead end for these apps.** The vendored xcframework ships a device-only (ios-arm64) slice, so it will not even link for the simulator, and there is no camera there regardless. Every iteration is a signed device build.

**Observability lives in the native console, not Dart.** When the app "just closes," the real cause shows up only via `xcrun devicectl device process launch --console --terminate-existing --device <UDID> <bundleId>`, not as a Dart exception. Wire native device-console capture before writing feature code. **Caveat learned the hard way: in a release build, Dart `print`/`debugPrint` does NOT reliably reach that console — only native logs (the SDK's own `NSLog`/`os_log`) do.** So you cannot profile or inspect from Dart on a release device build. Surface any diagnostics you need — per-stage timings, tensor shapes, the raw first detection box, the camera buffer WxH — on the **UI/HUD** instead. On PyroGuard a single on-screen debug line (`prev=1280x720 buf=720x1280 sensor=90 raw0=...`) is what finally pinned the orientation bug.

**Use release builds on device.** Debug mode hangs on launch on recent iOS/Xcode combinations ("Xcode is taking longer than expected to start debugging"), which means weaker logs and no hot reload exactly when the hard bug appears. Tapping a *debug* build's home-screen icon shows a "debug mode apps can only be launched from Flutter tooling" screen — that is expected, not a bug. Release builds launch standalone from the icon and run inference at full speed; prefer them on device.

**macOS can silently revoke the terminal's file access (TCC).** Mid-project, every flutter/git command in the project dir started failing with `Operation not permitted` — even `ls` and `getcwd` — because the project lived under `~/Desktop` and the terminal app lost Desktop access after an OS update/restart. `cd`/`pwd` kept working (shell builtins read `$PWD`), which masked the real cause. Fix: grant the terminal app **Full Disk Access** (System Settings → Privacy & Security → Full Disk Access) and fully relaunch it, or keep projects outside the protected `~/Desktop`/`~/Documents`/`~/Downloads` folders.

---

## 6. Per-app SPEC TEMPLATE

The orchestrator fills this out completely before any worker starts. A worker runs dark between gates, so a gap here becomes a guess there. No section may be left as "TBD" when handed off.

```
# SPEC: <AppName>

## One-line pitch
<what it does, who it is for>

## Model
- Source (HF repo / origin):
- Architecture:
- Melange model name:
- Melange version:
- Input tensor: dtype[shape], layout (NCHW/NHWC), value range, normalization
- Output tensor: dtype[shape], semantic layout of each dimension
- Post-processing baked into ONNX? (e.g. is NMS included or not)
- Classes / labels:
- modelMode to use and why:

## Input source
- Camera / mic / file
- Pixel format or sample rate requested
- Orientation handling required (e.g. portrait device, landscape buffer)

## Pre-processing pipeline (ordered, exact)
1. ...

## Post-processing pipeline (ordered, exact)
1. ...
(Be explicit about coordinate spaces, letterbox inverse, sigmoid-or-not,
 per-class vs global NMS, thresholds.)

## UI
- Left to the worker. Spec states only the functional must-haves
  (e.g. "live overlay of results with confidence", "per-class live count",
  "inference latency readout"). The worker chooses the visual design.

## Platform targets
- iOS minimum, Android minSdk
- Known OS traps for this model/artifact

## Validation focus
- The specific correctness traps most likely for THIS model
  (see VALIDATION.md Tier A) that the worker must cover with tests.
```

### Worked example (PyroGuard), for reference

```
# SPEC: PyroGuard

## One-line pitch
Real-time fire and smoke detector for industrial-safety prospects.

## Model
- Source: leeyunjai/yolo11-firedetect
- Architecture: YOLO11s fine-tuned (fire/smoke)
- Melange model name: ajayshah/FireDetectionYOLO
- Melange version: 1
- Input: float32[1,3,640,640], NCHW, values 0.0-1.0 (divide by 255)
- Output: float32[1,6,8400]; per anchor [cx, cy, w, h, fire_conf, smoke_conf];
  coords in 640x640 space; 8400 anchors across 80/40/20 grids
- NMS baked in? No. Implement in pure Dart.
- Classes: ["fire", "smoke"]
- modelMode: RUN_AUTO. (No modelMode avoided the FP32-GPU crash — that was fixed
  by ZETIC filtering GPU server-side for iOS 26.3+. Post-fix, selection serves
  TFLITE_FP16/CPU (~383 ms); RUN_AUTO will pick a Neural-Engine artifact
  automatically if/when ZETIC serves one.)

## Input source
- Rear camera, cheapest usable pixel format
- Device held portrait. On this iOS setup the BGRA buffer is delivered UPRIGHT
  (buf=720x1280), so the model sees the scene correctly with NO rotation.
  Verify the actual buffer WxH per device/format — do not assume landscape. The
  bug we hit was the overlay applying a *spurious* 90 degree rotation, not a
  missing one. (Android YUV420 buffers may differ; measure, don't assume.)

## Pre-processing
1. Capture frame bytes
2. Letterbox-resize to 640x640 (pad 0.5), preserving aspect
3. BGR -> RGB if needed
4. Normalize /255.0
5. Reorder to NCHW [1,3,640,640]
6. Flatten to Float32List, wrap as Tensor.float32List

## Post-processing
1. For each of 8400 anchors read [cx,cy,w,h,fire,smoke]
2. Keep where max(class scores) > threshold (default 0.25)
3. cxcywh -> x1y1x2y2
4. Undo letterbox (exact reverse of pre-processing) into screen space
5. Per-class NMS, IoU 0.45
6. Emit Detection{bbox,label,conf}

## UI
- Worker's choice; must show boxes+conf, per-class count, latency readout.

## Platform targets
- iOS 16.6+, Android minSdk 24
- Trap: FP32-GPU CoreML artifact crashes in MPSGraph on iOS/macOS 26.3+. Not
  client-fixable (no modelMode avoids it); fixed by ZETIC filtering GPU
  server-side. Always read the *served* artifact from the native console and
  confirm it is not GPU on affected OS versions.

## Validation focus
- Letterbox inverse round-trip; channel-major [1,6,8400] decode;
  per-class (not global) NMS; orientation correctness (verify the real buffer
  orientation on-device — the overlay can both under- and over-rotate).
```

---

## 7. Checkpoints

Work runs autonomously between gates; each agent stops and waits for the human at each gate. The four gates — GATE 0 (Melange upload, after Stage 0 exploration) through GATE 3 (device handoff) — are defined in AGENTS.md. The worker's `HANDOFF.md` living ticket is created at the start of the worker phase (after GATE 1) and finalized at GATE 3. Stage 0 model exploration is in EXPLORATION.md. The validation battery each worker must clear before the final gate is in VALIDATION.md.