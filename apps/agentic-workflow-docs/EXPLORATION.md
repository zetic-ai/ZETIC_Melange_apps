# EXPLORATION.md — Stage 0: Model Discovery & Export

This is the **first** stage of the workflow, before any spec or app code. It turns a
target technology into ready-to-upload model artifacts, so the human's only manual
step is a dashboard drag-and-drop. It ends at **GATE 0**.

Read CLAUDE.md and AGENTS.md first; this stage feeds them.

---

## 1. What this stage produces

For each target use-case, a populated app folder under `apps/<AppName>/` containing:

- `export.py` — the exact, re-runnable export recipe used.
- `<model>.onnx` — the exported model artifact.
- `sample_input.npy` — a sample input tensor, correct shape + dtype (random values
  are fine; Melange only needs shape/dtype to compile — see §7).
- `melange_upload.md` — the dashboard handoff: exactly what the human drags in and
  registers, and exactly what to paste back (§6 template).
- `model_selection.md` — the top-5 shortlist, the scoring, and why the winner was
  chosen (so the decision is auditable, not a black box — §8 template).
- A pre-drafted SPEC stub (CLAUDE.md section 6) with everything the ONNX already
  reveals filled in, and the Melange name/version fields as **late-binding
  placeholders** — the locally-verified ONNX contract already fixes shapes and
  dtypes; only the registered name/version arrive with the GATE-0 paste-back.

The Explorer does **not** write app code. That is the worker's job — and since
v2.0 the worker starts in parallel with the GATE-0 upload (see AGENTS.md).

---

## 2. The same-technology batch rule (binding, for now)

One exploration run targets **one technology family** — all YOLO, or all OCR, or all
TTS, or all audio tagging. The orchestrator spins off one Explorer agent per
**use-case** within that family (e.g. for YOLO: fire/smoke, PPE, weapons,
crowd-density, license-plate). Each agent owns one use-case and one app folder.

Why homogeneous:
- **Export is per-architecture-family.** There is no universal ONNX exporter. A batch
  of one family reuses a single export recipe (e.g. the Ultralytics YOLO recipe);
  mixing families would need a different recipe per agent and defeat the parallelism.
- **Melange-fit reasoning stays consistent** across the batch (same op set, same
  shape conventions, same known traps).

If a future run needs mixed families, that is a new design conversation — keep batches
single-family until then.

---

## 3. The Explorer agent's process (ordered)

1. **Receive assignment** from the orchestrator: the technology family + one specific
   use-case + target sector context (who the demo is for). The Explorer is deployed
   **into its own dedicated git worktree from the start** — before it searches, before
   it reasons, even if this is only a feasibility/plausibility pre-check that may end in
   NO-GO. Every agent is worktree-isolated from deployment (see AGENTS.md, "Every agent
   runs in its own worktree"); an unused worktree is cheap and can always be deleted, so
   there is no reason to skip it even for a read-only pre-check.
2. **Search Hugging Face.** Use the Hub API (`huggingface_hub.HfApi().list_models(...)`)
   filtered by the task tag for the family, plus free-text for the use-case, sorted by
   downloads/likes/recency. Pull a wide candidate set, then narrow.
3. **Build a top-5 shortlist**, scored against the Melange-fit rubric (§4). Record the
   score and a one-line note per candidate in `model_selection.md`.
4. **Reason and pick one winner** — the model best suited to Melange + the use-case,
   not just the most popular. State the trade-off in `model_selection.md`.
5. **Export to ONNX** with the family's recipe: **static shapes** (`dynamic=False`),
   fixed input dims, a known-good opset (opset 12 worked for PyroGuard), no
   half-precision in the ONNX itself (Melange handles precision). Record the exact
   command in `export.py`.
6. **Generate `sample_input.npy`** — `np.random.rand(*shape).astype(dtype)` matching
   the model's declared input. Random values are acceptable (§7).
7. **Populate the app folder**: artifacts + `melange_upload.md` + `model_selection.md`
   + the pre-drafted SPEC stub.
8. **Return and stop at GATE 0**, stating plainly what it needs from the human (the
   `melange_upload.md` is that statement). It must name the model, the file names, and
   the exact fields it needs pasted back.

The Explorer runs dark between assignment and GATE 0, exactly like a worker between
gates. It never guesses a registered model name — that stays a late-binding
placeholder until the paste-back — but the pipeline no longer waits at GATE 0
either: spec finalization and the worker build proceed in parallel with the upload,
on the strength of the locally-verified ONNX contract (see §9 and AGENTS.md).

---

## 4. Melange-fit scoring rubric

Score each candidate; the winner is the best balance, not the top of any single column.

- **Exportable to ONNX.** Has a `.pt`/`.onnx`/loadable checkpoint, or a known export
  path for its family. No export path → disqualified, however good the model.
- **Static-shape friendly.** Fixed input dims; clean fixed-shape ONNX export. Dynamic
  axes and control flow fight Melange's compile step.
- **Standard ops.** Conventional architectures convert cleanly. Exotic/custom ops,
  unusual attention variants, or non-standard layers risk a failed conversion. (Note:
  the iOS-26 MPSGraph *GPU* crash is an OS-side bug handled server-side by ZETIC — it
  is NOT a selection criterion; do not reject a model over it.)
- **Mobile-sized.** Parameter count / file size sane for on-device (think single-digit
  to low-tens of MB after conversion, not multi-hundred-MB LLM-scale).
- **License.** Permits commercial/demo use — this supports ZETIC's GTM. Flag
  restrictive licenses explicitly.
- **Quality + popularity signals.** Downloads, likes, recency, real eval numbers, a
  maintained repo. Popularity is a tiebreaker, not the goal.
- **Task fit.** Genuinely solves the assigned use-case, with documented I/O.
- **Output format known.** You can describe each output dimension and whether
  post-processing (e.g. NMS) is baked in. Undocumented output = a downstream guess.

---

## 5. Export recipes (a per-family library)

There is no universal exporter — maintain one recipe per architecture family, keyed by
family, and reuse it across that family's batch.

- **YOLO (Ultralytics):** the PyroGuard `export.py` is the reference recipe —
  `YOLO(path).export(format='onnx', imgsz=640, opset=12, simplify=True,
  dynamic=False, half=False)`.
- **Others (OCR, TTS, audio tagging, diarization, etc.):** add the recipe the first
  time that family is explored, then reuse. Each recipe records: load, export call,
  opset, input shape/dtype, and how `sample_input.npy` is shaped.

When a family has no recipe yet, writing it is part of that exploration run.

---

## 6. `melange_upload.md` template (the human's GATE-0 instructions)

```
# Melange upload — <AppName>

Drag these into the dashboard:
- model:  <file>.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/<ModelName>
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  <dtype>[shape], <layout>
- output tensor: <dtype>[shape], <semantic layout>
- classes / labels: <...>

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (the build is already running in parallel; this unblocks
only the name/version injection and the device run):
- the model name + version you registered
  (the dashboard header shows "ZETIC | <Name>" — that "ZETIC |" is the
   org/workspace DISPLAY prefix, NOT the account; the SDK name is
   ajayshah/<Name> WITH the slash)
  (the dashboard does NOT echo a version — the first upload is version 1,
   confirmed at the first SDK create())
- the served input/output shapes the dashboard shows (used to RECONCILE against
  the spec; a mismatch is stop-the-line for that app)
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)
```

---

## 7. Hard-won realities for exploration

- **`sample_input.npy` is random noise of the right shape.** That is fine for Melange,
  which only needs shape + dtype to compile. It does NOT validate that the model
  produces sensible outputs — that is the on-device run (VALIDATION.md Tier C). Never
  imply the model is "validated" because it converted.
- **"Benchmarked" is not "deployable"** (carries over from CLAUDE.md section 5): a fast
  dashboard row may never be served for a given chip. Exploration cannot promise NPU
  speed; it can only promise a clean, convertible artifact.
- **Static shapes or bust.** Export with fixed dims; confirm the ONNX has no dynamic
  axes before handing off.
- **License is a real gate**, not a footnote — a non-commercial model can sink a
  GTM demo. Surface it in `model_selection.md`.
- **The agent cannot upload to Melange.** There is no confirmed programmatic
  upload/conversion endpoint; the dashboard step is human-owned by design. If ZETIC
  later ships an upload API, GATE 0 can be automated — until then it is a hard human
  gate.

### Vision models: validate hard before you commit, then curate the best images

`sample_input.npy` random noise only proves shape/dtype for the Melange COMPILE — it
does NOT validate that the model produces correct outputs. For ANY vision model, before
committing to it, HEAVILY validate behavior: run the ACTUAL exported ONNX (via
onnxruntime) with the EXACT preprocessing the app will use, against a
GROUND-TRUTH-LABELED test set, and measure real accuracy — not just "it converts / it's
popular / the card looks good".

- **Prefer VALIDATION-GATED SELECTION.** When the choice matters, test the top
  candidates HEAD-TO-HEAD on one shared labeled set (parallel, isolated workspaces) and
  crown the winner on proven accuracy, not shortlist reasoning. Reject degenerate/weak
  models — e.g. a classifier that never predicts one of its classes, or a detector whose
  real recall is near chance. (This run's fundus model looked fine on paper but never
  predicted "healthy" — only measured validation caught it, and only after a 6-way
  bakeoff did a real winner emerge.)
- **If the model IS good, curate the 2–3 best demo images**: real images the model
  handles CORRECTLY and CONFIDENTLY, selected by measured output (not eyeballed), with
  rendered overlays/viz so the human can verify — so the live demo shows the model at its
  best. If the model is weak, surface it LOUDLY (aggregate metrics + caveats, not a
  highlight reel) and swap it before building.

This is the exploration-stage complement to VALIDATION.md — do it before GATE 0, not
after, so a weak model is caught before any app is built on it.

---

## 8. `model_selection.md` template (auditable decision record)

```
# Model selection — <AppName> (<technology family>, use-case: <...>)

## Shortlist (top 5)
| Rank | HF repo | Downloads | License | Export path | Melange-fit notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1    | ...     | ...       | ...     | ...         | ...               | ...   |
| ...  |         |           |         |             |                   |       |

## Winner: <repo>
Why this one over the runners-up (Melange-fit + task-fit trade-off, in 2-4 lines):
- ...

## Export
- Recipe: <family recipe / export.py>
- Input:  <dtype>[shape], <layout>, value range / normalization
- Output: <dtype>[shape], <semantic layout>, post-processing baked in? (y/n)
- Opset, static shapes confirmed: <...>
```

---

## 9. GATE 0 — Model ready for Melange upload

The Explorer presents the populated folder and stops. The human:
1. Uploads per `melange_upload.md`, registers the model, waits for READY.
2. Pastes back the registered model name + version, the served input/output shapes,
   and the modelMode (default RUN_AUTO).

**What GATE 0 does and does not gate (v2.0).** The Explorer exported the ONNX
locally and verified it in onnxruntime, so the model CONTRACT — input/output
shapes, dtypes, normalization, output semantics — is already ground truth at
export time. The paste-back supplies only what the pipeline genuinely cannot know
until the upload happens: the REGISTERED model name and version (plus confirmation
of the served shapes). Spec finalization and the worker build therefore proceed
**in parallel** with the upload, against late-binding name/version placeholders
(see AGENTS.md); GATE 0 gates only the name/version injection into the app's
constants file and the physical device run.

Two dashboard traps to keep in view:
- The dashboard header shows `ZETIC | <Name>`. That `ZETIC |` is the org/workspace
  DISPLAY prefix, not the account — the SDK name is `ajayshah/<Name>` with the
  slash (see CLAUDE.md section 5).
- The dashboard does NOT echo a version. The first upload is version 1; confirm it
  at the first SDK `create()`.

When the paste-back arrives, the orchestrator RECONCILES the served shapes against
the spec — a reconciliation step now, not a prerequisite. A mismatch is a
stop-the-line event for that app (rare; the locally-verified ONNX is almost always
right). On a clean reconcile, the registered name/version are injected as a
one-file, one-commit change. The registered name may legitimately differ from the
Explorer's proposal (it has, twice); that is an accepted, cheap rename, not an
error.

GATE 0 remains a hard human gate: the agent cannot upload to Melange (§7), and the
human alone owns the dashboard and the paste-back.
