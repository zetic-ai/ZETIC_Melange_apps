#!/usr/bin/env python3
"""Generate the Tier-A T10 golden fixture for the PAD preprocessor.

Replicates `buildPadInput` (lib/services/preprocessor.dart) EXACTLY — the same
half-pixel-center bilinear resize, BGR channel order, [0,255] range (NO /255),
and NCHW layout — on a small deterministic synthetic BGR image, and writes the
source + expected output to test/fixtures/pad_golden.json. The Dart test rebuilds
the same source and asserts its output matches, proving the Dart preprocessing
reproduces the reference numerics cross-language.

Run: melange-env/bin/python apps/LiveGate/tools/gen_golden.py
"""
import json
import math
import os

SIZE = 80  # kPadInputSize


def synth_bgr(w, h):
    """Deterministic BGR gradient/checker, row-major, 3 bytes/px."""
    px = []
    for y in range(h):
        for x in range(w):
            b = (x * 13 + y * 7) % 256
            g = (x * 5 + y * 11 + 40) % 256
            r = (x * 3 + y * 17 + 90) % 256
            px += [b, g, r]
    return px


def sample(bgr, w, h, x, y):
    x0 = math.floor(x)
    y0 = math.floor(y)
    fx = x - x0
    fy = y - y0
    x1 = x0 + 1
    y1 = y0 + 1

    def clampx(v):
        return 0 if v < 0 else (w - 1 if v >= w else v)

    def clampy(v):
        return 0 if v < 0 else (h - 1 if v >= h else v)

    def px(xx, yy, c):
        return bgr[(clampy(yy) * w + clampx(xx)) * 3 + c]

    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    out = []
    for c in range(3):
        v = (px(x0, y0, c) * w00 + px(x1, y0, c) * w10 +
             px(x0, y1, c) * w01 + px(x1, y1, c) * w11)
        # Match Dart num.round(): round half AWAY from zero. All v >= 0 here.
        v = math.floor(v + 0.5)
        out.append(0 if v < 0 else (255 if v > 255 else v))
    return out  # [b, g, r]


def build_pad_input(bgr, w, h, x1, y1, x2, y2):
    ix1, iy1, ix2, iy2 = map(math.floor, (x1, y1, x2, y2))
    cw = ix2 - ix1 + 1
    ch = iy2 - iy1 + 1
    area = SIZE * SIZE
    out = [0.0] * (3 * area)
    for dy in range(SIZE):
        sy = iy1 + (dy + 0.5) * ch / SIZE - 0.5
        for dx in range(SIZE):
            sx = ix1 + (dx + 0.5) * cw / SIZE - 0.5
            b, g, r = sample(bgr, w, h, sx, sy)
            idx = dy * SIZE + dx
            out[idx] = float(b)
            out[area + idx] = float(g)
            out[2 * area + idx] = float(r)
    return out


def main():
    # A 24x18 source, cropped to a sub-rect that upsamples to 80x80 (exercises
    # bilinear in both axes, non-integer scale).
    w, h = 24, 18
    bgr = synth_bgr(w, h)
    crop = (2.0, 1.0, 20.0, 15.0)  # x1,y1,x2,y2 inclusive far edge
    expected = build_pad_input(bgr, w, h, *crop)

    out = {
        "size": SIZE,
        "srcWidth": w,
        "srcHeight": h,
        "srcBgr": bgr,
        "crop": {"x1": crop[0], "y1": crop[1], "x2": crop[2], "y2": crop[3]},
        "expectedNchw": expected,
        "note": "BGR channel order (0=B,1=G,2=R), range [0,255] (NOT /255), NCHW.",
    }
    dst = os.path.join(
        os.path.dirname(__file__), "..", "Flutter", "test", "fixtures",
        "pad_golden.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f)
    print("wrote", os.path.relpath(dst))
    print("max value in golden:", max(expected), "(proves range is [0,255], not [0,1])")


if __name__ == "__main__":
    main()
