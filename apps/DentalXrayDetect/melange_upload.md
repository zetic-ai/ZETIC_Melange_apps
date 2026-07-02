# Melange upload — DentalXrayDetect

Drag these into the dashboard:
- model:  dentalxray-yolo11n.onnx
- sample: sample_input.npy

Create the model with:
- name:    ajayshah/DentalXRayDetect   (the SDK requires the fully-qualified account/project form:
                              account `ajayshah`, project `DentalXRayDetect`. A bare project name fails
                              on-device with MlangeException(3): "Model name must include account name and
                              project name separated by slash(/)". Match case EXACTLY: capital R in "XRay",
                              which differs from the folder name DentalXrayDetect.)
- version: 1

Verify after upload (the dashboard should echo these back):
- input tensor:  float32[1, 3, 640, 640], NCHW, RGB, values 0.0–1.0 (pixels / 255)
- output tensor: float32[1, 7, 8400], channel-major. Per anchor: [cx, cy, w, h, s0, s1, s2]
  = 4 box coords (pixel space in the 640×640 letterbox frame) + 3 class scores.
  Class scores are ALREADY sigmoid-activated (0–1) in-graph; NMS is NOT baked in.
- classes / labels: 0 caries, 1 periapical_lesion, 2 impacted_tooth

Then: trigger benchmark, wait for CONVERTING -> OPTIMIZING -> READY.

Paste back to the agent (it is BLOCKED until you do):
- the model name + version you registered
- the served input/output shapes the dashboard shows
- modelMode: default RUN_AUTO
  (Do NOT use RUN_ACCURACY as a crash workaround — it isn't one. The iOS/macOS
  26.3+ CoreML-GPU crash is handled server-side by ZETIC filtering the GPU
  candidate; no client mode avoids it. See CLAUDE.md section 5.)

---
## ⚠️ License flag for the human (read before any commercial use)
The winning checkpoint `liodon-ai/dental-panoramic-detector` is **CC-BY-NC-4.0
(NON-COMMERCIAL)**. It is fine for an internal capability-proof demo, but it CANNOT be
shipped in a commercial product as-is. If this demo advances toward a commercial GTM
build, swap to the MIT-licensed drop-in fallback documented in `model_selection.md`
(`Sentoz/dental-opg-cavity-detection-model`) — same YOLO export recipe, imgsz 640 — and
re-register. This is a legal gate, not a technical one.

## ⚠️ Clinical framing (do not overclaim)
On-device deployment changes DATA RESIDENCY only (radiographs/PHI stay in the operatory).
It does NOT confer or change any FDA clearance. These are public research weights trained
on PANORAMIC radiographs — a capability proof, not a cleared diagnostic device.
