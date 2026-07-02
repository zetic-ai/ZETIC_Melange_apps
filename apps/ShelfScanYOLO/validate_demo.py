"""
ShelfScanYOLO — demo-image inference + accuracy validation (re-runnable).

Reproduces the EXACT ShelfScanYOLO pre/post-processing on the exported ONNX and
scores predictions against SKU-110K ground-truth boxes.

Model:  shelfscan-yolo11s-sku110k.onnx  (YOLO11s, trained on SKU-110K, 1 class "object")
  Input : float32[1,3,640,640], NCHW, RGB, /255, LETTERBOXED (pad 114-gray).
  Output: float32[1,5,8400], channel-major -> per anchor [cx,cy,w,h,score].
          Box coords are in 640x640 LETTERBOX pixel space.
          The single class score is ALREADY sigmoid'd in-graph -> do NOT re-apply sigmoid.
          NMS is NOT baked in.

Usage:
  # Score the selected demo images against their stored GT and (re)draw overlays:
  python validate_demo.py --score demo_images/demo_gt.json

  # Run inference on any image (no GT), just draw the overlay:
  python validate_demo.py --image path/to/shelf.jpg
"""
import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(HERE, "shelfscan-yolo11s-sku110k.onnx")

CONF_THRES = 0.25   # class-score threshold (spec start value)
IOU_NMS = 0.45      # NMS IoU (spec)
IOU_MATCH = 0.50    # pred<->GT match IoU for recall/precision
INPUT_SZ = 640
PAD_VALUE = 114     # gray letterbox pad (matches Ultralytics /255 -> ~0.447)


# ----------------------------------------------------------------------------- preprocessing
def letterbox(img_rgb, new_size=INPUT_SZ, pad_value=PAD_VALUE):
    """Resize preserving aspect ratio, pad to square. Returns (padded, scale, pad_x, pad_y)."""
    h, w = img_rgb.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), pad_value, dtype=np.uint8)
    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(img_rgb):
    """img_rgb: HxWx3 uint8 RGB -> (tensor[1,3,640,640] float32, meta)."""
    padded, scale, pad_x, pad_y = letterbox(img_rgb)
    x = padded.astype(np.float32) / 255.0            # 0..1
    x = np.transpose(x, (2, 0, 1))                   # HWC -> CHW
    x = np.expand_dims(x, 0)                          # NCHW
    return np.ascontiguousarray(x), (scale, pad_x, pad_y)


# ----------------------------------------------------------------------------- postprocessing
def decode(output, meta, conf_thres=CONF_THRES):
    """output: [1,5,8400] channel-major. Returns xyxy boxes in ORIGINAL image px + scores."""
    scale, pad_x, pad_y = meta
    pred = output[0]                     # (5, 8400)
    assert pred.shape[0] == 5, f"expected channel-major (5,N), got {pred.shape}"
    cx, cy, w, h, score = pred          # each (8400,)  -- score already sigmoid'd in-graph

    keep = score > conf_thres           # threshold BEFORE geometry
    cx, cy, w, h, score = cx[keep], cy[keep], w[keep], h[keep], score[keep]

    # cxcywh (640 letterbox px) -> xyxy (640 letterbox px)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # undo letterbox: subtract pad, divide by scale -> original image px
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes, score


def nms(boxes, scores, iou_thres=IOU_NMS):
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty((0,))
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.clip(xx2 - xx1, 0, None)
        ih = np.clip(yy2 - yy1, 0, None)
        inter = iw * ih
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    keep = np.array(keep, dtype=int)
    return boxes[keep], scores[keep]


# ----------------------------------------------------------------------------- inference
_SESSION = None


def get_session():
    global _SESSION
    if _SESSION is None:
        _SESSION = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    return _SESSION


def infer(img_rgb, conf_thres=CONF_THRES, iou_thres=IOU_NMS):
    sess = get_session()
    inp = sess.get_inputs()[0].name
    x, meta = preprocess(img_rgb)
    out = sess.run(None, {inp: x})[0]
    boxes, scores = decode(out, meta, conf_thres)
    boxes, scores = nms(boxes, scores, iou_thres)
    return boxes, scores


# ----------------------------------------------------------------------------- scoring vs GT
def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> IoU (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def score_against_gt(pred_boxes, pred_scores, gt_boxes, iou_match=IOU_MATCH):
    """Greedy 1-1 match preds->GT at IoU>=iou_match. Returns dict of metrics."""
    gt_boxes = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    if n_pred == 0:
        return dict(n_gt=n_gt, n_pred=0, tp=0, fp=0,
                    recall=0.0, precision=0.0, mean_conf=0.0)
    ious = iou_matrix(pred_boxes, gt_boxes)                # (n_pred, n_gt)
    order = np.argsort(pred_scores)[::-1]                  # high conf first
    gt_taken = np.zeros(n_gt, dtype=bool)
    tp = 0
    for pi in order:
        row = ious[pi].copy()
        row[gt_taken] = -1
        gj = int(np.argmax(row))
        if row[gj] >= iou_match:
            gt_taken[gj] = True
            tp += 1
    fp = n_pred - tp
    recall = tp / n_gt if n_gt else 0.0
    precision = tp / n_pred if n_pred else 0.0
    return dict(n_gt=n_gt, n_pred=n_pred, tp=tp, fp=fp,
                recall=recall, precision=precision,
                mean_conf=float(np.mean(pred_scores)))


# ----------------------------------------------------------------------------- overlay render
def draw_overlay(img_rgb, pred_boxes, pred_scores, gt_boxes=None, out_path=None):
    """Predicted boxes in GREEN (+conf), GT boxes in RED (thin). Saves BGR PNG."""
    canvas = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()
    H, W = canvas.shape[:2]
    t = max(1, int(round(min(H, W) / 500)))
    if gt_boxes is not None:
        for (x1, y1, x2, y2) in np.asarray(gt_boxes).reshape(-1, 4):
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 0, 255), max(1, t - 1))
    for (x1, y1, x2, y2), s in zip(pred_boxes, pred_scores):
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 220, 0), t)
    # headline count
    label = f"{len(pred_boxes)} products detected"
    fs = min(H, W) / 900
    cv2.rectangle(canvas, (0, 0), (int(14 * len(label) * fs) + 20, int(50 * fs) + 20),
                  (0, 0, 0), -1)
    cv2.putText(canvas, label, (10, int(40 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), max(1, t))
    if out_path:
        cv2.imwrite(out_path, canvas)
    return canvas


# ----------------------------------------------------------------------------- CLI
def _load_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", help="JSON of {filename: {gt_boxes_xyxy: [...]}} to validate")
    ap.add_argument("--image", help="single image to run + overlay (no GT)")
    ap.add_argument("--conf", type=float, default=CONF_THRES)
    ap.add_argument("--iou", type=float, default=IOU_NMS)
    args = ap.parse_args()

    if args.image:
        img = _load_rgb(args.image)
        boxes, scores = infer(img, args.conf, args.iou)
        out = os.path.splitext(args.image)[0] + "_overlay.png"
        draw_overlay(img, boxes, scores, out_path=out)
        print(f"{args.image}: {len(boxes)} detections, "
              f"mean_conf={np.mean(scores) if len(scores) else 0:.3f} -> {out}")
        return

    if args.score:
        with open(args.score) as f:
            gt = json.load(f)
        base = os.path.dirname(os.path.abspath(args.score))
        rows = []
        for fname, rec in gt.items():
            path = os.path.join(base, fname)
            img = _load_rgb(path)
            boxes, scores = infer(img, args.conf, args.iou)
            m = score_against_gt(boxes, scores, rec["gt_boxes_xyxy"], IOU_MATCH)
            out = os.path.join(base, os.path.splitext(fname)[0] + "_overlay.png")
            draw_overlay(img, boxes, scores, rec["gt_boxes_xyxy"], out_path=out)
            rows.append((fname, m))
            print(f"{fname}: GT={m['n_gt']} det={m['n_pred']} "
                  f"recall={m['recall']:.3f} prec={m['precision']:.3f} "
                  f"mean_conf={m['mean_conf']:.3f} -> {os.path.basename(out)}")
        if rows:
            rec = np.array([r[1]['recall'] for r in rows])
            pre = np.array([r[1]['precision'] for r in rows])
            print(f"\nAGGREGATE over {len(rows)} selected: "
                  f"median recall={np.median(rec):.3f}, median precision={np.median(pre):.3f}")
        return

    ap.error("pass --score <gt.json> or --image <path>")


if __name__ == "__main__":
    main()
