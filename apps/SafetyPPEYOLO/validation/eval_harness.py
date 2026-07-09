"""Head-to-head ONNX eval of PPE candidates on the shared GT test set.
Preprocessing = EXACT app pipeline: letterbox to 640x640 (gray pad 0.5), /255, NCHW.
Decode: [1, 4+nc, 8400] channel-major, sigmoid already applied, conf 0.25, per-class NMS IoU 0.45.
Match: greedy IoU >= 0.5 per class -> per-class precision/recall.
"""
import json, os, time
import numpy as np
import cv2
import onnxruntime as ort
from collections import defaultdict

SCRATCH = "/private/tmp/claude-501/-Users-ajayshah-Desktop-ZETIC-ZETIC-Melange-apps/6abc40ad-72cf-4e76-bb63-6290bbef3255/scratchpad"
TS = f"{SCRATCH}/ppe_testset"
CONF, NMS_IOU, MATCH_IOU = 0.25, 0.45, 0.50
CLASSES = ["helmet", "no-helmet", "vest", "no-vest", "person"]

MODELS = {
    "safetyvision_v2s": (f"{SCRATCH}/ppe_onnx/safetyvision_v2s.onnx",
        {3: "helmet", 7: "no-helmet", 12: "vest", 9: "no-vest", 11: "person"}),
    "hansung_n": (f"{SCRATCH}/ppe_onnx/hansung_n.onnx",
        {0: "helmet", 2: "no-helmet", 7: "vest", 4: "no-vest", 5: "person"}),
    "leeyunjai_11s": (f"{SCRATCH}/ppe_onnx/leeyunjai_11s.onnx",
        {0: "helmet", 1: "no-helmet", 4: "vest", 2: "no-vest", 3: "person"}),
    "hexmon_m": (f"{SCRATCH}/ppe_onnx/hexmon_m.onnx",
        {3: "helmet", 8: "no-helmet", 13: "vest", 10: "no-vest", 11: "person"}),
    "melih_11n": (f"{SCRATCH}/ppe_onnx/melih_11n.onnx",
        {0: "helmet", 2: "no-helmet", 3: "vest", 1: "person"}),
    "badmuriss_8s": (f"{SCRATCH}/ppe_onnx/badmuriss_8s.onnx",
        {0: "helmet", 1: "vest"}),
}

def letterbox(img, size=640, pad=0.5):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad * 255.0, dtype=np.float32)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized.astype(np.float32)
    return canvas, r, dx, dy

def preprocess(path):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lb, r, dx, dy = letterbox(rgb)
    x = (lb / 255.0).transpose(2, 0, 1)[None].astype(np.float32)
    return x, r, dx, dy

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def nms_per_class(dets):
    out = []
    for c in set(d[5] for d in dets):
        cl = sorted([d for d in dets if d[5] == c], key=lambda d: -d[4])
        keep = []
        for d in cl:
            if all(iou(d[:4], k[:4]) < NMS_IOU for k in keep):
                keep.append(d)
        out += keep
    return out

def decode(raw, mapping, r, dx, dy, W, H):
    a = raw[0]  # [4+nc, 8400]
    nc = a.shape[0] - 4
    boxes = a[:4]           # cx cy w h in 640 space
    scores = a[4:]          # [nc, 8400] already sigmoid
    cls_id = scores.argmax(0)
    cls_sc = scores.max(0)
    keep = cls_sc >= CONF
    dets = []
    for i in np.where(keep)[0]:
        cid = int(cls_id[i])
        if cid not in mapping:
            continue
        cx, cy, w, h = boxes[:, i]
        x1 = (cx - w / 2 - dx) / r; y1 = (cy - h / 2 - dy) / r
        x2 = (cx + w / 2 - dx) / r; y2 = (cy + h / 2 - dy) / r
        x1, y1 = max(0, min(W, x1)), max(0, min(H, y1))
        x2, y2 = max(0, min(W, x2)), max(0, min(H, y2))
        dets.append([x1, y1, x2, y2, float(cls_sc[i]), mapping[cid]])
    return nms_per_class(dets)

gt = json.load(open(f"{TS}/ground_truth.json"))
results = {}
per_image_dets = {}

for name, (onnx_path, mapping) in MODELS.items():
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    img_dets = {}
    t0 = time.time()
    for rec in gt:
        x, r, dx, dy = preprocess(f"{TS}/{rec['file']}")
        raw = sess.run(None, {inp: x})[0]
        dets = decode(raw, mapping, r, dx, dy, rec["w"], rec["h"])
        img_dets[rec["file"]] = dets
        for c in CLASSES:
            gts = [b["xyxy"] for b in rec["boxes"] if b["cls"] == c]
            preds = sorted([d for d in dets if d[5] == c], key=lambda d: -d[4])
            used = set()
            for p in preds:
                best, bi = 0.0, -1
                for gi, g in enumerate(gts):
                    if gi in used: continue
                    v = iou(p[:4], g)
                    if v > best: best, bi = v, gi
                if best >= MATCH_IOU:
                    tp[c] += 1; used.add(bi)
                else:
                    fp[c] += 1
            fn[c] += len(gts) - len(used)
    dt = (time.time() - t0) / len(gt)
    per_image_dets[name] = img_dets
    res = {}
    for c in CLASSES:
        P = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else float("nan")
        R = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else float("nan")
        res[c] = {"tp": tp[c], "fp": fp[c], "fn": fn[c], "P": P, "R": R}
    tps, fps, fns = sum(tp.values()), sum(fp.values()), sum(fn.values())
    res["_micro"] = {"P": tps / (tps + fps) if tps+fps else 0, "R": tps / (tps + fns) if tps+fns else 0}
    res["_ms_per_img"] = dt * 1000
    results[name] = res
    print(f"\n== {name} ==  ({dt*1000:.0f} ms/img CPU ORT)")
    for c in CLASSES:
        r_ = res[c]
        print(f"  {c:10s} TP={r_['tp']:>3} FP={r_['fp']:>3} FN={r_['fn']:>3}  P={r_['P']:.3f} R={r_['R']:.3f}")
    print(f"  micro     P={res['_micro']['P']:.3f} R={res['_micro']['R']:.3f}")

json.dump(results, open(f"{SCRATCH}/ppe_eval_results.json", "w"), indent=1, default=float)
json.dump(per_image_dets, open(f"{SCRATCH}/ppe_eval_dets.json", "w"), indent=1, default=float)
print("\nsaved ppe_eval_results.json / ppe_eval_dets.json")
