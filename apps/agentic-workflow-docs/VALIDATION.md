# VALIDATION.md — The Quality Gate

## Philosophy

For these apps the ML is roughly 30% of the risk. The other 70% is integration surface: vendor SDK opacity, OS and hardware fragmentation, signing and deploy friction, observability, and pipeline plumbing. Almost none of that 70% is visible until the app runs on a physical device, which an agent cannot do.

So validation is triaged into three tiers. A worker may only claim "ready for device," never "done," and only after Tier A passes and Tier B is satisfied. Tier C is surfaced to the human, not tested.

---

## Tier A — Autonomous gates (must pass before GATE 3)

All of these run without a device. A worker is not ready for handoff until every one is green.

### A1. Static analysis
- `flutter analyze` returns zero errors and zero warnings.
- No unused code paths, no `// TODO`, no stub functions left in shipped files.

### A2. Build
- The app compiles for the target platform (device build config, release mode where the SDK requires it). A compile that only succeeds in debug does not count if release is the device path.
- **Launcher icon.** A custom, domain-identifying launcher icon is present (not the default Flutter icon), generated for iOS + Android via `flutter_launcher_icons`.
- **Product name.** A cool, domain-identifying product name is set as the user-facing display name (iOS `CFBundleDisplayName`, Android `android:label`, in-app title) — not "Runner" or the raw package name. Bundle id, folder, and Melange model name stay unchanged.

### A3. Unit tests (correctness traps)

These traps do not throw when wrong; they produce plausible-but-wrong output that is nearly impossible to catch in a live demo. Each must be covered by a test with hand-constructed data and known expected output.

- **Tensor layout.** Decode against a hand-built output tensor containing exactly one known result. Assert the worker reads channel-major vs row-major correctly (for example [1,6,8400] is channel-major: stride across anchors, not across the 6).
- **Coordinate space.** Test pixel-space (0-N) vs normalized (0-1) handling with known values. A wrong assumption shifts every box.
- **Letterbox / resize inverse.** Round-trip a known box: forward letterbox, then inverse, assert it returns to the original within tolerance. The inverse must be the exact reverse order of the forward steps.
- **Orientation.** Assert the input buffer matches the model's expected orientation before inference — which may mean *no* rotation. Do not assume the camera buffer is landscape: on the PyroGuard iOS setup it arrived upright (720x1280) and the real bug was the overlay applying a *spurious* rotation, transposing every wide box into a tall sliver. Easy to miss because the overlay can rotate boxes back (or over-rotate them) and hide the error. The reliable check is to log the actual buffer WxH and one known box on-device and confirm the drawn box matches the object; a pure-Dart test can at least assert the chosen transform round-trips a known box for the orientation you believe you have.
- **Score semantics.** Test whether class scores need a sigmoid (or other activation) applied, matching this model. Wrong activation silently degrades confidence.
- **Suppression semantics.** For detectors: test per-class vs global NMS explicitly. Two overlapping boxes of different classes must both survive per-class NMS.
- **Threshold behavior.** Test the confidence-threshold boundary (just-below is dropped, just-above is kept).

Adapt the list to the task: an audio tagger has no letterbox but has windowing, sample-rate, and frame-overlap traps that map one-to-one onto the same idea (hand-built input, known expected output).

### A4. Hot-path micro-benchmark
- A Dart benchmark (in `test/benchmark/`) feeds mock tensors of the real shape through the full pure-Dart hot path (preprocess plus decode plus suppression) and reports median time over many iterations.
- This is the only latency number an agent can produce honestly. It is the post-processing budget, not the end-to-end device latency.
- Record the median. It becomes the baseline for Tier B.

---

## Tier B — Optimization checklist (the 0.5% rule)

Real end-to-end latency cannot be measured without the device, because the NPU time is fixed by Melange and only appears on hardware. What a worker controls and can measure is the pure-Dart hot path from A4.

**The rule:** every optimization the worker applies must produce a measurable improvement on the A4 micro-benchmark of at least 0.5% of the post-processing budget, demonstrated by a before/after number. An optimization that cannot show that delta is removed, because it only adds complexity. The worker logs each one with its measured delta.

Work through this checklist and apply (or justify skipping) each lever:

**Isolate and copy cost**
- Avoid spawning a fresh isolate per frame. Reuse a long-lived isolate, or run inline where the SDK binds the model handle to one isolate. Minimize the bytes copied across the isolate boundary (a full-frame copy twice per frame is a large hidden tax).

**Preprocessing**
- Pre-allocate the input `Float32List` once and write in place; do not allocate per frame.
- Fuse resize, normalize, and NCHW reorder into a single pass over the pixels rather than several passes with intermediate buffers.
- Operate on typed-data views, not boxed `List`s. Avoid per-pixel Dart-level loops where a bulk operation exists.
- Request the cheapest usable camera pixel format to minimize conversion work.

**Decode**
- Iterate the anchors once. Apply the confidence threshold before computing box geometry, so rejected anchors cost almost nothing.
- Do not allocate a result object per anchor until after the threshold passes.

**Suppression**
- Sort once. Pre-compute box areas. Bucket per class. Threshold before the O(n^2) step so n is small.

**Frame flow**
- Guard with a `_busy` flag so frames do not pile up; drop frames rather than queue them. Throttle inference to device capability only where needed, not by default on fast silicon.

**Model lifecycle**
- Warm the model with one dummy inference right after load, so the first real frame is not the slow one.
- Ensure the model is cached and not re-downloaded on every launch.

**Render**
- Repaint the overlay only when detections change; do not rebuild the painter every frame for nothing.

---

## Tier C — Human-handoff runtime-risk checklist (surfaced, not tested)

These are the device-only realities an agent cannot verify or fix. The worker fills this out at GATE 3 so the human runs the device with eyes open. This is the honest 70%.

- **Served artifact.** State the expected served backend and precision, and flag that the client cannot force it. Read the *actual* served `target`+`apType` from the native console (e.g. `runtimeApType=CPU`) — that, not the requested mode, is the truth. Call out any known crash path (FP32-GPU CoreML on iOS/macOS 26.3+) and remember the realistic non-crashing fallback is often CPU (TFLITE_FP16, ~hundreds of ms), not the NPU. Getting onto the Neural Engine is a separate ask to ZETIC.
- **modelMode.** State which mode is set. Do NOT claim a modelMode avoids a crashing artifact: on PyroGuard all four modes returned the same crashing FP32-GPU artifact. The iOS-26 GPU crash was fixed *server-side* by ZETIC filtering the GPU candidate for that OS, not by a client mode. If a new OS crashes in MPSGraph, escalate to ZETIC to filter GPU for it.
- **Native observability.** Give the exact device-console command to watch during the run (`xcrun devicectl device process launch --console --terminate-existing --device <UDID> <bundleId>`), because Dart will show nothing on a native crash. **Note that Dart `print`/`debugPrint` does not surface in this console on a release build — only native logs do** — so on-device diagnostics (timings, shapes, buffer WxH, raw boxes) must be shown on the UI/HUD, not logged from Dart. Recommend wiring this before the first run.
- **Signing and OS gates.** List the manual, non-scriptable gates: signing identity, Developer Mode, "Always Allow", minimum OS. These are pure tax but block the run.
- **Build config.** State the build mode to use on device (release where debug hangs) and why.
- **Network and cold start.** The model downloads on first launch over the network; on poor conference Wi-Fi that is a spinner. Recommend pre-download or pre-warm, and a fresh-install rehearsal.
- **Non-determinism acceptance.** Server-side selection can return a different artifact minute to minute, and the backend itself can change under you (on PyroGuard, a re-target made one launch run at ~400 ms, then minutes later the same build resolved to the crashing GPU artifact again; separately, the GPU-filter fix flipped behavior server-side with no app change). "It ran once" is not evidence. Acceptance is: runs cleanly across multiple cold starts and at least one fresh install before it counts as demo-ready — and re-verify after any backend/model re-target.
- **Secrets.** Note that the personal key is embedded in the client.