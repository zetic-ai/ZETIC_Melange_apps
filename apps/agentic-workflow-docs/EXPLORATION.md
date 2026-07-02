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
  reveals filled in, and the GATE-0 fields (Melange name/version, served shapes)
  left blank for the human to confirm.

The Explorer does **not** write app code. That is the worker's job, after GATE 0.

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
   use-case + target sector context (who the demo is for).
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
gates — it does not push past GATE 0 with guessed model names or shapes.

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

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
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

Those paste-back values are the only things the pipeline genuinely cannot know until
the upload happens. With them, the orchestrator finalizes the SPEC and proceeds to
GATE 1. Until then, the app is BLOCKED at GATE 0.
