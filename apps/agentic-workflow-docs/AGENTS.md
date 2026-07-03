# AGENTS.md — Orchestration Protocol

**Protocol version: Orchestrator v2.0.** Changelog: v1 = serial gates, the Melange upload blocked the build; v2.0 = the worker build runs in parallel with GATE 0 via late-binding model constants, and GATE 1/2 approvals are orchestrator-held.

Delegation here is manual and deliberate. The orchestrator decides when to hand off, what to hand off, and when to resume. Nothing spawns automatically.

---

## Agent roster

| Agent | Model / effort | Owns | Writes app code? |
|---|---|---|---|
| Master orchestrator | Opus 4.8, maximum reasoning effort | Specs, delegation, gates, review (no worktree — owns the main tree) | No |
| Explorer (one per use-case) | Opus 4.8, high reasoning effort | Stage 0: find + export one model, one folder — in its own worktree | No |
| Worker (one per app) | Opus 4.8, high reasoning effort | One app, one branch, one worktree | Yes |

Explorers and workers are separate roles, and **every agent of either role runs in its
own dedicated git worktree from the moment it is deployed** (see "Every agent runs in
its own worktree" below and "Branch / worktree mechanics"). A run's Explorers all target
the **same technology family** (see EXPLORATION.md). The orchestrator may let an
Explorer's session continue as that app's worker once the GATE-0 upload package is
handed to the human (the build runs in parallel with the upload), or spawn a fresh
worker — either is fine, since each agent already has its own branch/worktree.

Set the model and reasoning effort per the Claude Code workflows docs (code.claude.com/docs/en/workflows). Confirm the exact invocation there; the intent is "max-effort planner, high-effort builders." Do not assume flag names; verify them.

---

## Every agent runs in its own worktree (binding)

**Every agent — Explorers, workers, feasibility/plausibility pre-checks, doc-editors, and any one-off task agent — runs in its own dedicated git worktree from the second it is deployed**, before it has done any work, even while it is still only *reasoning about whether a task is plausible or feasible*. A feasibility pre-check that may well conclude NO-GO still starts in its own worktree. No agent ever works in the main repo working tree, and no agent ever touches another agent's worktree. Setting up the worktree is part of deploying the agent (the one bit of git plumbing the orchestrator does inline, because it is what CREATES the isolation).

**Why always isolate:** worktree isolation is cheap and fully reversible — an unused or unchanged worktree can always be deleted (git auto-removes unchanged ones), so there is **zero downside to over-isolating** and a real downside to sharing (collisions, half-work polluting main, the confusion of shared trees). Default to always giving an agent its own worktree; never skip it because the task "seems read-only" or "might not go anywhere." The orchestrator may delete the worktree afterward if the agent produced nothing worth keeping.

---

## The orchestrator's loop

**Delegate everything task-specific (absolute).** The orchestrator's main loop does **zero** concrete work. Every task-specific action — HuggingFace search, ONNX export, code edits, builds, research, doc edits (including edits to these very orchestration docs), validation runs, git-artifact wrangling, even a "quick" one-line check — is done by a spawned sub-agent, never inline. The main loop ONLY decides, delegates, reviews returned work, and holds gates. There is no "too small to delegate": small size is not a reason to inline — if it is task work, it goes to an agent. The one exception is the git plumbing that CREATES agent isolation (worktree setup, see "Branch / worktree mechanics") and the human-gated dashboard step.

1. **Stage 0 (exploration).** Pick one technology family and the target use-cases for this run. Spin off one Explorer per use-case (see EXPLORATION.md). Each searches Hugging Face, picks the best model for Melange, exports it, and populates its app folder with the ONNX, `sample_input.npy`, `melange_upload.md`, `model_selection.md`, and a pre-drafted spec stub. All Explorers in a run share one technology family (one export recipe). Because the Explorer verifies the exported ONNX locally in onnxruntime, the model CONTRACT — input/output shapes, dtypes, normalization, output semantics — is already ground truth at export time; the spec does not wait for the dashboard.
2. Finalize the per-app spec from the Explorer's stub and the local ONNX contract, so every section of the CLAUDE.md section 6 template is filled. The Melange model name/version fields are **late-binding placeholders** (`[LATE-BINDING — placeholder until GATE-0 paste-back]`, see CLAUDE.md section 6); everything else must be concrete. A gap here becomes a guess in a dark worker session.
3. **GATE 1 (spec approval):** the orchestrator reviews the finalized spec with full rigor and approves it itself (see "Gate ownership" below), reporting the decision to the human after the fact. Do not delegate to a worker until the spec is approved.
4. **GATE 0 (Melange upload) — runs in parallel with steps 5–7.** Each Explorer hands the human its `melange_upload.md`. The human drags the two artifacts into the dashboard, registers the model, waits for READY, and pastes back the registered model name + version and the served input/output shapes. GATE 0 no longer blocks spec finalization or the worker build; it gates ONLY (a) injecting the registered name/version into the app's late-binding constants file and (b) the physical device run. **Accepted risk, stated plainly:** the registered name may differ from the proposal (it did, twice: proposals DocTextDetector/SceneTextRecognizer registered as LiveDocRedact_Detect/SignTranslate_Rec). The human ensures registration is sane; even a mismatch is a cheap one-constant rename. Note the dashboard does NOT echo a version — first upload = version 1, confirmed at the first SDK `create()`.
5. Hand the approved spec to a worker on a fresh branch **immediately — do not wait for the paste-back**. The worker's **first action** is to create `HANDOFF.md` — the living Jira ticket — from the spec, with the build plan as its Todo list: mostly `[ ]` open, `[x]` only for already-done Stage-0 export artifacts, and `[ ] [BLOCKED – owner]` for anything it cannot resolve from the app side (e.g. the pending Melange paste-back). The worker builds against a single dedicated constants file (e.g. `lib/services/model_registry.dart`) holding the Melange model name/version as clearly-marked placeholders; nothing else in the app may reference the name or version, so "plugging in the model" is a one-file change plus one commit. Everything below the model boundary — the entire pure-Dart pipeline, Tier A tests (hand-built mock tensors), UI, benchmarks, icon/naming, even the `--no-codesign` release build — is buildable with zero model bytes, because Melange downloads the model at runtime anyway. The worker then plans and proposes its test list.
6. **GATE 2 (approach approval):** the worker returns a build plan plus the Tier A test list it intends to write, and presents this initial `HANDOFF.md` ticket (the plan-of-record) alongside them, before writing app code. The orchestrator reviews and approves or redirects (see "Gate ownership").
7. Worker runs dark: writes the app, writes the tests, runs the validation loop until the Tier A battery passes and the Tier B optimization checklist is satisfied, keeping `HANDOFF.md` updated throughout (flipping `[ ]`→`[x]` as tasks complete, updating blocked items).
8. **Paste-back reconciliation.** When the GATE-0 paste-back arrives, the orchestrator reconciles the served input/output shapes against the spec — served-shape verification is a reconciliation step now, not a prerequisite. A mismatch is a stop-the-line event for that app (rare; the locally-verified ONNX is almost always right). On a clean reconcile, the registered name/version are injected into the constants file: one file, one commit.
9. **GATE 3 (handoff for device run):** the worker finalizes the living `HANDOFF.md` and returns the validation report plus the Tier C runtime-risk checklist, plus the paste-back reconciliation status (name/version injected, or still placeholders). The human performs the physical-device run (which requires the injection). The worker never claims "done"; it claims "ready for device."
10. Human reports device results. If a device-only issue appears, the orchestrator decides whether it is a worker fix (Dart pipeline) or a human/dashboard action (artifact retarget, OS trap).

Multiple Explorers and workers may be at different gates at the same time. The orchestrator tracks each app's gate state (GATE 0 through GATE 3).

---

## Gate ownership (v2.0)

The orchestrator holds gate approvals by default. It **reviews and self-approves GATE 1 (spec) and GATE 2 (approach), and accepts GATE-3 handoffs**, with full review rigor: read the actual artifacts (not the agent's summary of them), hold CLAUDE.md section 5 as fact, and verify the claims that matter at the boundary. Self-approval is not rubber-stamping — it is the same review, minus the human interrupt. Every gate decision is reported to the human transparently after the fact.

The human is interrupted only for:
- the GATE-0 dashboard upload and paste-back,
- physical device runs,
- secrets (the personal key at build time),
- merges / PR approval (agents raise PRs open, never merge),
- dropping or adding an app mid-run,
- any genuinely undecidable product call.

---

## Orchestrator lessons (hard-won)

- **Ground truth lives at the real boundary.** The running system / SDK / device is authoritative — above a confident human claim, a model card, or a green dashboard. Don't encode an unverified claim into docs, specs, or memory until the boundary that enforces it confirms it (the model-name format was wrong twice and only the on-device SDK error settled it).
- **Validate before you choose, not after.** Fan out candidates and test head-to-head on a shared ground-truth set before committing; selection quality is bounded by how empirically you compare.
- **Convertible → accurate → served → demo-ready are separate claims.** Never let a downstream claim borrow upstream credibility. "Benchmarked" (dashboard) ≠ "served" (device console apType) ≠ "demo-ready". Read the served artifact on the device, not the dashboard row.
- **Verify agent reports independently.** An agent's final message is a claim, not a fact — cheaply confirm the ones that matter (files exist, the specific line is right, the secret isn't staged, the PR is actually OPEN). Reviewing is the orchestrator's job; it is not the same as doing the work.
- **Gate the irreversible and the human-only; run dark in between.** Checkpoint where only a human can act or a wrong guess is expensive to unwind. A gap in a spec becomes a guess in every dark worker, and an early error multiplies across parallel branches.
- **Irreversible/outward actions need their own explicit go.** Merge, force-push, external send — don't upgrade a reversible action into an irreversible one "for completeness". Know the undo first; prefer revert over history-rewrite on shared branches.
- **Make a reproducible ground-truth harness the contract.** Downstream implementations must reproduce it EXACTLY, including the boring details (interpolation, thresholds, normalization, label order) — that's where silent wrongness hides (nearest-vs-bilinear resize nearly shipped).
- **Reward loud honesty.** Design asks so agents surface aggregate/honest metrics and caveats, not a highlight reel — a loud "this is weak and here's why" is what lets you make the real call.
- **Parallelize the independent; respect shared bottlenecks.** Isolate workspaces so parallel agents never collide, but serialize around true single resources (one device, one machine, one dashboard-human) and set expectations honestly.
- **Keep an always-current state map; prove liveness non-invasively.** For multi-entity parallel work, maintain an explicit per-entity gate/state table and confirm background work is alive via artifacts/mtimes, not by disrupting it.
- **The process is a tool, not dogma — bend it deliberately and say so** (this run intentionally broke the one-family rule and fast-tracked a gate; deviations were explicit, not silent).

---

## Branch / worktree mechanics

**Each agent gets its own worktree from deployment** — not just per-app workers, but Explorers, feasibility/plausibility pre-checks, doc-editors, and any one-off task agent. The orchestrator creates the worktree as it deploys the agent, so every agent has its own working directory and branch before it does any work and parallel agents never collide.

```bash
# from the repo root, one per agent — created as part of deploying that agent
git worktree add ../pyroguard-wt      -b app/pyroguard        # a per-app worker
git worktree add ../audiotag-wt       -b app/audiotagger      # another worker
git worktree add ../explore-plate-wt  -b explore/plate        # an Explorer (even a NO-GO pre-check)
git worktree add ../docs-enforce-wt   -b docs/enforce         # a one-off doc-editor agent
# ...one worktree per agent, whatever its role
```

Launch one agent session per worktree directory. Each agent:
- works only inside its own worktree,
- commits to its own branch,
- never touches another agent's worktree or files.

Because worktree isolation is cheap and fully reversible, always give an agent its own worktree — never skip it because the task "seems read-only" or "might not go anywhere." An unused or unchanged worktree can always be deleted (git auto-removes unchanged ones), so the orchestrator can tear it down afterward if the agent produced nothing worth keeping.

The orchestrator and its agents RAISE PRs but NEVER merge to `main`. After a successful human device run, open a PR (`gh pr create`, left OPEN) and stop — the human reviews and merges. The **PR body mirrors the app's `HANDOFF.md` ticket verbatim** (the same Goal / Todo List / Deliverables / References), optionally plus the standard "🤖 Generated with Claude Code" footer — one source of truth kept in sync, not a separately-authored PR summary, so a reviewer reads the same plan-of-record on the PR and in the repo. Because the PR body is this verbatim mirror, `HANDOFF.md` must itself be valid GitHub-Flavored Markdown (see "Handoff ticket format") — `##` section headers and `- [ ]`/`- [x]` task-list items that render as real checkboxes on GitHub — so mirroring it yields a clean, well-formed PR rather than a wall of Jira plaintext. "Raise/open a PR" means create it open, full stop; it never implies merging. Treat merging, force-pushing, or pushing to a shared branch as outward, hard-to-reverse actions that require an explicit human go each time, even mid-flow.

---

## Handoff contract (what an agent returns at each gate)

**At GATE 0 (Explorer):**
- The populated app folder: `export.py`, `<model>.onnx`, `sample_input.npy`.
- `melange_upload.md` — the exact dashboard steps plus the fields the human must paste back (model name/version, served shapes, modelMode default RUN_AUTO).
- `model_selection.md` — the top-5 shortlist, scoring, and winner rationale.
- A pre-drafted SPEC stub with the Melange name/version fields as late-binding placeholders (the locally-verified ONNX contract already fixes shapes, dtypes, and output semantics).
- The Explorer presents these and stops. The human does the dashboard upload — in parallel with the worker build, which does not wait for it; see EXPLORATION.md.

**At GATE 2:**
- A short build plan (files, pipeline approach, threading model).
- The late-binding constants plan: the single dedicated file (e.g. `lib/services/model_registry.dart`) that will own the Melange model name/version as clearly-marked placeholders until the GATE-0 paste-back — and confirmation that nothing else in the app references them.
- The exact Tier A test list it will write.
- The initial `HANDOFF.md` living ticket (Jira format below), created from the spec — the plan-of-record — with the build plan as its Todo list.
- Any spec ambiguity it found (this is the worker's one chance to ask before going dark).

**At GATE 3:**
- Validation report: Tier A results (analyze, build, unit tests, micro-benchmark numbers).
- Tier B optimization log: each optimization applied, with its measured delta on the Dart hot-path micro-benchmark, or a justification for skipping.
- Tier C runtime-risk checklist (from VALIDATION.md), filled for this app: served-artifact expectation, modelMode chosen, device-console command to watch, signing/build-config notes, network/cold-start risk, and the "run it N times" acceptance note.
- Paste-back reconciliation status: whether the GATE-0 paste-back has arrived, whether the served shapes reconciled cleanly against the spec, and whether the registered name/version are injected into the constants file or still placeholders (the device run requires the injection; a shape mismatch is stop-the-line).
- A custom, domain-identifying launcher icon (not the default Flutter icon), generated for iOS + Android via `flutter_launcher_icons` from a 1024x1024 source.
- A cool, domain-identifying product name set as the user-facing display name (iOS `CFBundleDisplayName`, Android `android:label`, in-app title), distinct from the model/folder name (bundle id, folder, and Melange model name unchanged).
- The finalized `HANDOFF.md` — the already-existing living ticket updated to its GATE-3 state (completed items marked `[x]`, blocked items updated), not authored fresh here.

The worker presents these and stops. The human runs the device.

---

## Handoff ticket format (HANDOFF.md)

The worker creates `HANDOFF.md` in the project folder as its **first build artifact** — a living plan-of-record — right after GATE 1, using the strict GitHub-Flavored Markdown structure below. It keeps it updated through the build (flipping `- [ ]`→`- [x]` as tasks complete) and finalizes it at GATE 3. This keeps every app's handoff paste-ready into the real tracker from the moment building starts, and forces the worker to state plainly what is done, what is blocked, and what the human must do next. The Todo list uses `- [x]` for completed, `- [ ]` for open, and `- [ ] **[BLOCKED – owner]**` for anything the worker cannot resolve from the app side (for example a server-side artifact issue). Blocked items must name the root cause and the owner.

`HANDOFF.md` MUST be valid GitHub-Flavored Markdown, because it is mirrored verbatim into the PR body and has to render cleanly on GitHub — not Jira plaintext. The four sections are H2 headers, each followed by a blank line: `## Goal`, `## Todo List`, `## Deliverables`, `## References`. Include the test device when known (as a `## References` bullet). The rules are strict:

- `## Goal`: a prose paragraph.
- `## Todo List`: GitHub **task-list** items, each on ONE source line, each led by a `- ` list marker with a single space inside the brackets so GitHub renders a **real checkbox**: `- [x] completed item`, `- [ ] open item`, and `- [ ] **[BLOCKED – owner]** root cause / what's blocked` (bold the BLOCKED tag). Never a bare `[x]`/`[ ]` (no marker) and never `[]` (no inner space) — those do NOT become checkboxes on GitHub. One task = one line: do NOT hard-wrap an item across multiple source lines, since GitHub renders single newlines in a PR/issue body as line breaks that fragment the item; it soft-wraps in the browser on its own.
- `## Deliverables` and `## References`: `- ` bullet lists, one item per source line.
- Exactly one blank line between sections and between a header and its content.
- Optional final line, after a blank line: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

This same ticket content is what goes in the app's PR body when the PR is raised (see "Branch / worktree mechanics") — the PR mirrors `HANDOFF.md` verbatim, not a separately-authored summary.

### Worked example (PyroGuard HANDOFF.md)

This example shows the ticket in its **finalized (GATE-3) state**; early in a build most Todo items are still `[ ]` open.

```markdown
## Goal

A real-time, fully on-device fire & smoke detection demo for Flutter (iOS), powered by a YOLO11s detector through the ZETIC Melange SDK. Streams the live camera feed, runs detection each frame on-device, and overlays labeled fire/smoke boxes with a live latency + detection-count HUD.

## Todo List

- [x] Create core Flutter structure (loading screen, camera screen, theme, HUD).
- [x] Export YOLO11s (leeyunjai/yolo11-firedetect) to ONNX (opset 12, 640x640) and register on Melange (ajayshah/FireDetectionYOLO, v1).
- [x] Melange lifecycle wrapper (create -> Tensor.float32List -> run -> close).
- [x] Preprocessing: letterbox 640x640, BGRA (iOS) / YUV420 (Android) decode, NCHW float32 normalization.
- [x] Post-processing: decode [1,6,8400] channel-major, threshold, un-letterbox, per-class NMS.
- [x] Detection overlay (rotation + BoxFit.cover mapping) and HUD.
- [x] iOS signing/deploy (team, NSCameraUsageDescription, iOS 16.6 min); run on a physical iPhone.
- [x] Resolve device-only xcframework (no simulator slice) via physical-device + release-mode builds.
- [x] Per-stage latency profiler (preprocess / run / postprocess) and device-console crash capture.
- [x] Inference-time crash (RESOLVED by ZETIC): Melange was serving a COREML_FP32/GPU artifact that aborted in Apple MPSGraph on iOS 26.3+ (MLIR pass manager failed, SIGABRT). No client modelMode avoided it — all four (AUTO/ACCURACY/SPEED/QUANTIZED) resolved to the same artifact. ZETIC filtered the GPU candidate server-side for affected OS versions; now serves TFLITE_FP16/CPU, no crash.
- [x] Fix accuracy: the camera buffer is already delivered upright (720x1280), so removed the SPURIOUS 90-degree overlay rotation that transposed every box into a tall sliver. No input rotation was needed.
- [ ] **[BLOCKED – ZETIC backend]** Inference latency (~400ms) is CPU-bound: the served TFLITE_FP16/CPU artifact benchmarks ~383ms; the Dart pipeline is only ~20ms. Real fix is a CoreML / Neural-Engine artifact from ZETIC (~3ms); runAuto will auto-select it once served. Minor secondary (worker-side): replace the per-frame compute() double-isolate spawn (~20ms).
- [ ] Android run verification once iOS is stable.
- [ ] Static sample-image validation harness for pre/post-processing.

## Deliverables

- Flutter source under FireDetectionYOLO/Flutter/ (screens, MelangeService, preprocessor, postprocessor, NMS, detection model, overlay/HUD).
- Model assets: export.py, firedetect-11s.onnx, sample_input.npy, registered Melange model (ajayshah/FireDetectionYOLO v1).
- iOS config: signing (team WVJ22PPYBP), Info.plist camera usage, Podfile (iOS 16.6, vendored ZeticMLange.xcframework).
- Diagnostics: HANDOFF.md (root-cause analysis, backend-selection test matrix, resume checklist), in-code latency profiler, devicectl console workflow.

## References

- App directory: apps/FireDetectionYOLO
- Core SDK: ZETIC Melange (zetic_mlange 1.8.1, Flutter FFI)
- Model: YOLO11s fire/smoke — leeyunjai/yolo11-firedetect (input float32[1,3,640,640], output float32[1,6,8400], classes: fire, smoke)
- Frameworks: Flutter, camera plugin, CoreML / Apple Neural Engine (via Melange), Ultralytics (export)
- Test device: iPhone 15 (iPhone15,4, A16), iOS 26.5
```