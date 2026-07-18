# Melange upload — ContentModeration

Drag these into the dashboard:
- model:  nsfw-vit-tiny-384.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/ContentModeration
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1,3,384,384], NCHW, RGB, normalized (x-0.5)/0.5 -> [-1,1]
- output tensor: float32[1,2] RAW LOGITS, order [NSFW, SFW] (index 0 = NSFW, index 1 = SFW)
- classes / labels: ["NSFW", "SFW"]

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
  candidate; no client mode avoids it. This is a ViT — attention heads are the
  exact fusion pattern that hit the MPSGraph GPU crash on PyroGuard, so if a new
  OS crashes at first inference, escalate to ZETIC to filter GPU for that OS.
  See CLAUDE.md section 5.)
