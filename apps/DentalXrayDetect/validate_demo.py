#!/usr/bin/env python3
"""
DentalXrayDetect — demo-image validation / scoring harness.

Reproduces the exact ONNX inference + post-processing from SPEC_STUB.md and
scores predictions against DENTEX ground-truth annotations.

Model:   dentalxray-yolo11n.onnx  (YOLO11n, liodon-ai/dental-panoramic-detector)
         Input  float32[1,3,640,640] NCHW RGB, pixels/255, letterboxed (pad 114).
         Output float32[1,7,8400] channel-major: [cx,cy,w,h,s0,s1,s2].
         Box coords are 640-letterbox PIXEL space. Class scores ALREADY sigmoid'd.
         NMS NOT baked in.
Classes: 0 caries, 1 periapical_lesion, 2 impacted_tooth.

GT:      DENTEX challenge validation split (validation_triple.json, 50 imgs, held-out).
         category_id_3: 0 Impacted, 1 Caries, 2 Periapical Lesion, 3 Deep Caries.
         Mapped to model classes: Caries/Deep Caries->0, Periapical->1, Impacted->2.

Usage:   python validate_demo.py                 # score all imgs, print summary
         python validate_demo.py --overlay IMG   # render overlay PNG for one image

License: DENTEX images are CC-BY-NC-SA-4.0 (non-commercial). Capability proof only.
"""
import os, sys, json, argparse
import numpy as np
import cv2
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "dentalxray-yolo11n.onnx")

CLASS_NAMES = {0: "caries", 1: "periapical_lesion", 2: "impacted_tooth"}
# DENTEX category_id_3 -> model class id
DENTEX_DIAG_TO_MODEL = {0: 2, 1: 0, 2: 1, 3: 0}  # Impacted, Caries, Periapical, DeepCaries
DENTEX_DIAG_NAME = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}

CONF_THRES = 0.45   # model-card recommended (SPEC_STUB): at 0.25 caries over-fires
IOU_NMS    = 0.45   # per-class NMS IoU (task instruction)
IOU_MATCH  = 0.50   # GT-match IoU threshold


# ----------------------------- pre-processing -----------------------------
def letterbox(img_rgb, new=640, pad=114):
    """Resize keeping aspect ratio, pad to new x new. Returns tensor input, scale r, (dw,dh)."""
    h, w = img_rgb.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), pad, dtype=np.uint8)
    dw, dh = (new - nw) // 2, (new - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def preprocess(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # decodes grayscale to 3ch BGR
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    canvas, r, dw, dh = letterbox(rgb, 640, 114)
    x = canvas.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]  # NCHW
    return np.ascontiguousarray(x), r, dw, dh, rgb.shape[:2]


# ----------------------------- post-processing ----------------------------
def nms(boxes, scores, iou_thres):
    """Standard NMS. boxes xyxy. Returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(0); h = (yy2 - yy1).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def decode(out, r, dw, dh, orig_hw, conf_thres=CONF_THRES, iou_nms=IOU_NMS):
    """out: [1,7,8400] channel-major. Returns list of dict(box xyxy orig-space, cls, conf)."""
    o = out[0]                      # [7,8400]
    boxes_lb = o[:4].T              # [8400,4] cx,cy,w,h in 640 letterbox px
    cls = o[4:7].T                  # [8400,3] already sigmoid'd
    conf = cls.max(1)
    cid = cls.argmax(1)
    m = conf > conf_thres
    boxes_lb, conf, cid = boxes_lb[m], conf[m], cid[m]
    if len(boxes_lb) == 0:
        return []
    # cxcywh -> xyxy (letterbox space)
    cx, cy, ww, hh = boxes_lb[:, 0], boxes_lb[:, 1], boxes_lb[:, 2], boxes_lb[:, 3]
    xyxy = np.stack([cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2], 1)
    # invert letterbox -> original pixel space
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dw) / r
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dh) / r
    H, W = orig_hw
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
    # per-class NMS
    dets = []
    for c in np.unique(cid):
        idx = np.where(cid == c)[0]
        keep = nms(xyxy[idx], conf[idx], iou_nms)
        for k in keep:
            j = idx[k]
            dets.append({"box": xyxy[j].tolist(), "cls": int(c), "conf": float(conf[j])})
    dets.sort(key=lambda d: -d["conf"])
    return dets


def iou_xyxy(a, b):
    xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
    xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------- GT loading ---------------------------------
def load_gt(json_path):
    d = json.load(open(json_path))
    id2name = {im["id"]: im["file_name"] for im in d["images"]}
    gt = {}  # file_name -> list of dict(box xyxy, model_cls, diag)
    for a in d["annotations"]:
        fn = id2name[a["image_id"]]
        x, y, w, h = a["bbox"]
        diag = a["category_id_3"]
        gt.setdefault(fn, []).append({
            "box": [x, y, x + w, y + h],
            "model_cls": DENTEX_DIAG_TO_MODEL[diag],
            "diag": diag,
        })
    return gt


# ----------------------------- scoring ------------------------------------
def score_image(sess, iname, img_path, gt_list, conf_thres=CONF_THRES):
    x, r, dw, dh, orig = preprocess(img_path)
    out = sess.run(None, {iname: x})[0]
    dets = decode(out, r, dw, dh, orig, conf_thres=conf_thres)
    # match each det to a GT of same model class, IoU>=IOU_MATCH, greedy by conf
    used = set()
    for det in dets:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_list):
            if j in used or g["model_cls"] != det["cls"]:
                continue
            i = iou_xyxy(det["box"], g["box"])
            if i >= IOU_MATCH and i > best_iou:
                best_iou, best_j = i, j
        det["tp"] = best_j >= 0
        det["match_iou"] = best_iou
        det["match_diag"] = gt_list[best_j]["diag"] if best_j >= 0 else None
        if best_j >= 0:
            used.add(best_j)
    return dets, orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgdir", default=os.path.join(HERE, "work_val"))
    ap.add_argument("--gt", default=os.path.join(HERE, "work_val_triple.json"))
    ap.add_argument("--conf", type=float, default=CONF_THRES)
    ap.add_argument("--overlay", default=None, help="render overlay for one file_name")
    ap.add_argument("--outdir", default=os.path.join(HERE, "demo_images"))
    args = ap.parse_args()

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    gt = load_gt(args.gt)

    if args.overlay:
        render_overlay(sess, iname, args.imgdir, gt, args.overlay, args.outdir, args.conf)
        return

    # aggregate over all images
    files = sorted(gt.keys())
    n_gt = {0: 0, 1: 0, 2: 0}
    n_tp = {0: 0, 1: 0, 2: 0}   # GT instances that got a matching detection
    n_pred = {0: 0, 1: 0, 2: 0}
    n_fp = {0: 0, 1: 0, 2: 0}
    per_img = {}
    for fn in files:
        path = os.path.join(args.imgdir, fn)
        if not os.path.exists(path):
            continue
        dets, orig = score_image(sess, iname, path, gt[fn], args.conf)
        per_img[fn] = dets
        for g in gt[fn]:
            n_gt[g["model_cls"]] += 1
        matched_gt = set()
        for d in dets:
            n_pred[d["cls"]] += 1
            if d["tp"]:
                n_tp[d["cls"]] += 1
            else:
                n_fp[d["cls"]] += 1

    print(f"\n=== DentalXrayDetect validation @ conf={args.conf} IoU_nms={IOU_NMS} IoU_match={IOU_MATCH} ===")
    print(f"Images tested: {len([f for f in files if os.path.exists(os.path.join(args.imgdir,f))])}")
    print(f"{'class':<18}{'GT':>5}{'TP':>5}{'recall':>9}{'preds':>7}{'FP':>5}{'precision':>11}")
    tot_gt=tot_tp=tot_pred=tot_fp=0
    for c in [0, 1, 2]:
        rec = n_tp[c] / n_gt[c] if n_gt[c] else 0
        prec = n_tp[c] / n_pred[c] if n_pred[c] else 0
        print(f"{CLASS_NAMES[c]:<18}{n_gt[c]:>5}{n_tp[c]:>5}{rec:>9.2f}{n_pred[c]:>7}{n_fp[c]:>5}{prec:>11.2f}")
        tot_gt+=n_gt[c]; tot_tp+=n_tp[c]; tot_pred+=n_pred[c]; tot_fp+=n_fp[c]
    print(f"{'TOTAL':<18}{tot_gt:>5}{tot_tp:>5}{(tot_tp/tot_gt if tot_gt else 0):>9.2f}{tot_pred:>7}{tot_fp:>5}{(tot_tp/tot_pred if tot_pred else 0):>11.2f}")

    # rank candidate images: high-conf correct detections, tight IoU
    print("\n=== Top candidate images (>=1 TP, ranked by best TP conf) ===")
    ranked = []
    for fn, dets in per_img.items():
        tps = [d for d in dets if d["tp"]]
        if tps:
            best = max(tps, key=lambda d: d["conf"])
            fp = sum(1 for d in dets if not d["tp"])
            classes = sorted(set(CLASS_NAMES[d["cls"]] for d in tps))
            ranked.append((best["conf"], fn, len(tps), fp, best, classes))
    ranked.sort(reverse=True)
    for conf, fn, ntp, nfp, best, classes in ranked:
        print(f"  {fn:<12} TPs={ntp} FPs={nfp} bestTP={CLASS_NAMES[best['cls']]}"
              f" conf={best['conf']:.3f} IoU={best['match_iou']:.2f} "
              f"diag={DENTEX_DIAG_NAME[best['match_diag']]:<16} classes={classes}")


# ----------------------------- overlay rendering --------------------------
def render_overlay(sess, iname, imgdir, gt, fn, outdir, conf_thres):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(imgdir, fn)
    dets, orig = score_image(sess, iname, path, gt.get(fn, []), conf_thres)
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    # GT in yellow
    for g in gt.get(fn, []):
        x1, y1, x2, y2 = [int(v) for v in g["box"]]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 255), 3)
        cv2.putText(bgr, f"GT:{DENTEX_DIAG_NAME[g['diag']]}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    # predictions in green (TP) / red (FP)
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        color = (0, 255, 0) if d["tp"] else (0, 0, 255)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 3)
        tag = f"{CLASS_NAMES[d['cls']]} {d['conf']:.2f}"
        if d["tp"]:
            tag += f" IoU{d['match_iou']:.2f}"
        cv2.putText(bgr, tag, (x1, min(orig[0] - 4, y2 + 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    out = os.path.join(outdir, fn.replace(".png", "_overlay.png"))
    cv2.imwrite(out, bgr)
    print(f"wrote {out}  (dets={len(dets)}, tp={sum(d['tp'] for d in dets)})")


if __name__ == "__main__":
    main()
