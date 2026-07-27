#!/usr/bin/env python3
"""End-to-end fidelity + separation check through the shipped ONNX models.

This is the Tier-A "golden fidelity" (PAD) and the GATE-3 "does the threshold
still separate after alignment" (FACE) evidence that the Dart pipeline cannot
produce on its own (Dart can't run the Melange models offline). It reproduces
the app's EXACT preprocessing in Python and runs the committed ONNX:

  PAD  : the 2.7x demo crops -> 80x80 BGR [0,255] (NOT /255) -> softmax[class 1].
         Expect live~1.000, spoof~0.000, and the /255 path saturated (~0).
  FACE : YuNet 5-point detect -> ArcFace similarity warp (same transform as
         lib/services/face_align.dart) -> 112x112 BGR [0,255] -> L2-norm cosine.
         Expect same-person >> 0.363 >> different-person.

The FACE step needs a 5-point detector (the app uses ML Kit on-device, which is
unavailable in Python); YuNet is the standard OpenCV stand-in for this exact
SFace model. It is fetched to a local cache on first run.

Run: melange-env/bin/python apps/LiveGate/tools/e2e_pipeline_check.py
"""
import os
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(__file__)
APP = os.path.join(HERE, "..")
DEMO = os.path.join(APP, "demo_images")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = os.path.join(HERE, ".cache_yunet.onnx")

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def pad_score(crop_png, divide255=False):
    """Reproduce the PAD branch: crop image -> 80x80 BGR [0,255] -> softmax[1]."""
    s = ort.InferenceSession(os.path.join(APP, "pad_minifasnet_v2.onnx"))
    im = Image.open(crop_png).convert("RGB").resize((80, 80))
    rgb = np.asarray(im).astype(np.float32)
    bgr = rgb[:, :, ::-1]
    if divide255:
        bgr = bgr / 255.0
    x = np.transpose(bgr, (2, 0, 1))[None]
    out = s.run(None, {s.get_inputs()[0].name: np.ascontiguousarray(x)})[0]
    return float(softmax(out[0])[1])


def _ensure_yunet():
    if not os.path.exists(YUNET_PATH):
        print("fetching YuNet detector to", os.path.relpath(YUNET_PATH))
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)


def face_embed(path):
    """YuNet 5-pt detect -> similarity warp to ArcFace template -> SFace L2 emb."""
    s = ort.InferenceSession(os.path.join(APP, "face_sface.onnx"))
    img = cv2.imread(path)  # BGR
    det = cv2.FaceDetectorYN.create(YUNET_PATH, "",
                                    (img.shape[1], img.shape[0]), 0.6, 0.3, 5000)
    _, faces = det.detect(img)
    if faces is None:
        raise RuntimeError("no face in " + path)
    f = faces[0]
    pts = f[4:14].reshape(5, 2)  # reye, leye, nose, rmouth, lmouth
    # Order by image x (image-left first), matching the Dart detector.
    eyes = sorted([pts[0], pts[1]], key=lambda p: p[0])
    mouth = sorted([pts[3], pts[4]], key=lambda p: p[0])
    ordered = np.array([eyes[0], eyes[1], pts[2], mouth[0], mouth[1]],
                       dtype=np.float32)
    # Partial-affine == similarity (scale+rotation+translation), the same class
    # of transform lib/services/face_align.dart fits by least squares.
    m, _ = cv2.estimateAffinePartial2D(ordered, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    aligned = cv2.warpAffine(img, m, (112, 112))  # BGR 112x112, [0,255]
    x = np.transpose(aligned.astype(np.float32), (2, 0, 1))[None]
    v = s.run(None, {s.get_inputs()[0].name: np.ascontiguousarray(x)})[0][0]
    return v / np.linalg.norm(v)


def main():
    print("=== PAD (liveness) — 2.7x crops, BGR [0,255], softmax[class 1] ===")
    live = pad_score(os.path.join(DEMO, "pad_live_crop.png"))
    spoof = pad_score(os.path.join(DEMO, "pad_spoof_crop.png"))
    live_bug = pad_score(os.path.join(DEMO, "pad_live_crop.png"), divide255=True)
    print(f"  live_crop  softmax[1] = {live:.3f}   (expect ~1.000, LIVE  >= 0.45)")
    print(f"  spoof_crop softmax[1] = {spoof:.3f}   (expect ~0.000, SPOOF <  0.45)")
    print(f"  live_crop /255 (BUG)  = {live_bug:.3f}   (saturated ~0 — proves feed [0,255])")

    print("\n=== FACE (1:1 match) — YuNet + ArcFace similarity warp, cosine ===")
    _ensure_yunet()
    a = face_embed(os.path.join(DEMO, "face_ref_personA.png"))
    pa = face_embed(os.path.join(DEMO, "face_probe_personA.png"))
    pb = face_embed(os.path.join(DEMO, "face_probe_personB.png"))
    same = float(a @ pa)
    diff = float(a @ pb)
    print(f"  same person (A vs A) cosine = {same:.3f}   -> {'MATCH' if same >= 0.363 else 'no-match'}")
    print(f"  diff person (A vs B) cosine = {diff:.3f}   -> {'MATCH' if diff >= 0.363 else 'no-match'}")
    print(f"  threshold 0.363 separates: {same >= 0.363 and diff < 0.363}")
    print(f"  (naive resize WITHOUT alignment gives diff~0.44 -> FALSE MATCH; "
          f"alignment is why the gate is trustworthy)")


if __name__ == "__main__":
    main()
