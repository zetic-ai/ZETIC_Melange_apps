# ZETIC SDK Escalation — SkyScout (AerialDetectYOLO) hangs during on-device init on Galaxy S24 GPU

**Filed by:** registry owner `ajayshah`
**Date:** 2026-06-30
**App:** SkyScout (`com.zetic.aerialdetect`)
**Model:** `ajayshah/AerialDetectYOLO` — YOLOv8s VisDrone
**modelKey:** `c8bd921e663c4efd89e511bd13df0ca5`

---

## Summary

SkyScout's model (`ajayshah/AerialDetectYOLO` — YOLOv8s VisDrone, input `[1,3,928,928]`,
output `[1,14,17661]`) **hangs during on-device init on the Samsung Galaxy S24**. The UI
stays stuck on "Downloading & optimizing model" indefinitely (5+ minutes, never completes).

The **same model runs perfectly on iPhone 15** (A16, CoreML/Metal backend). Backend
selection, model download, and decryption all succeed on the S24 — **the hang is on-device,
inside the TFLite GPU delegate initialization**, not a network/registry problem.

---

## Device

| Field | Value |
|---|---|
| Device model | Samsung Galaxy S24 / SM-S921N |
| SoC | Exynos 2400 |
| SoC model (selection key) | `s5e9945` |
| SoC vendor | Samsung |
| GPU | Samsung Xclipse (OpenCL) |
| OS | Android 16 |
| adb serial | `R3CX501W7FD` |

---

## Served artifact (captured live from logcat)

Backend selection request/response (`BackendSelectionClient`), selection mode **AUTO**,
match level **SOC_VENDOR**, **rank 1 = `TFLITE_FP16` / `ap_type = GPU`**:

```
D BackendSelectionClient: requestBody   "selection_mode": "AUTO",
D BackendSelectionClient: requestBody   "rank": 1,
D BackendSelectionClient: requestBody     "device_model": "SM-S921N",
D BackendSelectionClient: requestBody     "soc_vendor": "Samsung",
D BackendSelectionClient: requestBody     "soc_model": "s5e9945"
D BackendSelectionClient: requestBody     "model_key": "c8bd921e663c4efd89e511bd13df0ca5"

D BackendSelectionClient: responseBody   "status": "BENCHMARKED",
D BackendSelectionClient: responseBody   "selection_mode": "AUTO",
D BackendSelectionClient: responseBody   "match_level": "SOC_VENDOR",
D BackendSelectionClient: responseBody   "total": 10,
D BackendSelectionClient: responseBody   "candidate": {
D BackendSelectionClient: responseBody     "rank": 1,
D BackendSelectionClient: responseBody     "target": "TFLITE_FP16",
D BackendSelectionClient: responseBody     "ap_type": "GPU",
D BackendSelectionClient: responseBody     "metrics": { "latency_ns": 636781065, "snr_avg": 77.17, ... }
D BackendSelectionClient: responseBody   "plan_type": "pro_plus"
```

Download + decrypt of the served artifact succeed:

```
I BackendSelectionExecutor: Downloaded candidate: rank=1, target=ZETIC_MLANGE_TARGET_TFLITE_FP16, apType=GPU,
    localPath=.../c8bd921e663c4efd89e511bd13df0ca5/backend_selection/rank_1_TFLITE_FP16/aerialdetect_yolov8s_visdrone.ztc
```

Model identity confirmed on device (`ZeticMLangeModel`):

```
D ZeticMLangeModel:   modelKey=c8bd921e663c4efd89e511bd13df0ca5
D ZeticMLangeModel:       [0] name=images,  dtype=float32, rank=4, shape=[1, 3, 928, 928]
D ZeticMLangeModel:       [0] name=output0, dtype=float32, rank=3, shape=[1, 14, 17661]
```

---

## Root cause (evidenced)

The TFLite **GPU delegate** (`TfLiteGpuDelegateV2`, OpenCL) **stalls while
compiling/warming the heavy ~397-node, 928×928-input graph** on this SoC's GPU.

logcat reaches the delegate node-replacement line and then **goes completely silent** —
no error, no completion:

```
D [ZETIC_MLANGE]: GPU Delegate turned on
I tflite  : Created TensorFlow Lite delegate for GPU.
D [ZETIC_MLANGE]: GPU Delegate added to options
I tflite  : Replacing 397 out of 397 node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 1 partitions for subgraph 0.
   <-- last line from this thread; then silence -->
```

The model init thread stays **RUNNING, pinned at ~100% of one core**, and never returns.
Live `top` on the stalled process (PID 12959) after several minutes:

```
  TID USER      PR  NI VIRT  RES  SHR S[%CPU] %MEM     TIME+ THREAD     PROCESS
13004 u0_a816   20   0 104G 1.1G 181M R  100  16.6   1:43.31 Thread-4   com.zetic.aerialdetect
```

`Thread-4` (the mlange model thread) is state **R (running)**, ~100% CPU, with CPU TIME+
continuously accumulating and **no further logcat output after the "Replacing 397/397"
line**. The UI remains stuck on "Downloading & optimizing model" the whole time. Force-stop
is the only way out.

Model node count for the served FP16 graph: **397 nodes** (all 397 handed to the GPU
delegate). Input resolution 928×928 makes this a heavy graph to compile/warm on the GPU.

---

## Key contrast (isolates it to GPU + heavy model)

- **PlateHawk** (`ajayshah/VehiclePlateYOLO` — YOLOv8n, 640×640 input) on the **same device
  + same GPU** is also served `TFLITE_FP16` / GPU and **initializes fine**. → not a
  device/GPU-delegate-availability problem in general.
- **iPhone 15** (A16, CoreML/Metal backend) runs **SkyScout fine**. → not a model-correctness
  problem; the FP16/CoreML path is healthy.
- Therefore the failure is specifically **the heavy 928×928 / 397-node model on this SoC's
  TFLite GPU (`TfLiteGpuDelegateV2`, OpenCL) delegate** — the combination of heavy graph +
  Exynos 2400 (`s5e9945`) Xclipse GPU delegate compile/warm stalls.

---

## Client mitigation already tried (did NOT work)

Setting `modelMode` on the client had no effect — all variants still resolved to the GPU
candidate and stalled identically:

| `modelMode` tried | Result |
|---|---|
| `RUN_QUANTIZED` | still resolved to `TFLITE_FP16` / GPU → stalled |
| `RUN_SPEED`     | still resolved to GPU → stalled |
| `RUN_ACCURACY`  | still resolved to GPU → stalled |

This is consistent with only `modelMode` reaching the selector but **not being able to move
off the GPU candidate** for this SoC. The client cannot work around this — it needs a
server-side selection fix.

---

## The ask (server-side)

**Filter the GPU / `TFLITE_FP16`-GPU candidate out of backend selection for this SoC
(Exynos 2400 / `s5e9945`) for this model**, and serve a working alternative instead:

- Prefer **`TFLITE_FP16` / CPU** (benchmarked healthy on this device — see the on-device
  latency table; CPU FP16 completes), or
- an **NPU / QNN** target if one is validated for this SoC.

Because the selection request already carries the SoC (`soc_model: s5e9945`,
`match_level: SOC_VENDOR`), the filter is **server-side and SoC-scoped** — it will not
affect other devices, and it mirrors the earlier server-side fix where ZETIC filtered the
GPU candidate for the iOS/macOS 26.3+ MPSGraph crash.

**Targeting info for the fix:**

| Field | Value |
|---|---|
| Model name | `ajayshah/AerialDetectYOLO` (YOLOv8s VisDrone) |
| modelKey | `c8bd921e663c4efd89e511bd13df0ca5` |
| Candidate to filter | `target = TFLITE_FP16`, `ap_type = GPU` (rank 1) |
| SoC scope | Samsung Exynos 2400 / `soc_model = s5e9945` |
| Registry owner / contact | `ajayshah` |

---

## Contact / context

- Registry owner: **`ajayshah`**.
- The selection request already includes the SoC (`match_level = SOC_VENDOR`,
  `soc_model = s5e9945`), so a **server-side, SoC-scoped filter** of the GPU candidate for
  modelKey `c8bd921e663c4efd89e511bd13df0ca5` is sufficient.
- iPhone 15 unaffected; PlateHawk on the same S24 GPU unaffected — no broad regression risk.
