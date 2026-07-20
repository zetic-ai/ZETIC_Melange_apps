"""Generate the VoxScribe app icon (1024x1024, full-bleed for iOS masking).

Motif: two overlapping speaker speech-bubbles (Speaker 1 = cyan, Speaker 2 =
pink) with a blended overlap region (the diarization "overlap" class), and three
transcript lines in the front bubble. Reads as "two people talking / who-spoke-
when" — the app's pitch — and stays legible at home-screen size.

Run:  python tool/make_icon.py   ->   tool/voxscribe_icon.png (+ _fg for Android)
"""
from PIL import Image, ImageDraw
import math

S = 1024


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def rounded_bubble(size, fill, radius, tail="bl"):
    """A rounded-rect speech bubble (RGBA) with a small tail."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(size * 0.06)
    box = [pad, pad, size - pad, int(size * 0.74)]
    d.rounded_rectangle(box, radius=radius, fill=fill)
    # tail
    if tail == "bl":
        d.polygon([(int(size*0.26), int(size*0.70)),
                   (int(size*0.20), int(size*0.90)),
                   (int(size*0.44), int(size*0.71))], fill=fill)
    else:  # br
        d.polygon([(int(size*0.74), int(size*0.70)),
                   (int(size*0.80), int(size*0.90)),
                   (int(size*0.56), int(size*0.71))], fill=fill)
    return img


def build(bg=True):
    base = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    if bg:
        # diagonal indigo -> violet gradient
        top = (40, 33, 120)      # #282178
        bot = (124, 58, 237)     # #7C3AED
        grad = Image.new("RGBA", (S, S))
        gd = grad.load()
        for y in range(S):
            for x in range(0, S, 4):
                t = (x + y) / (2 * S)
                c = lerp(top, bot, t)
                for dx in range(4):
                    if x + dx < S:
                        gd[x + dx, y] = c + (255,)
        # soft radial highlight top-left
        hi = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        hd = ImageDraw.Draw(hi)
        for r in range(int(S*0.55), 0, -6):
            a = int(38 * (1 - r / (S*0.55)))
            hd.ellipse([int(S*0.30)-r, int(S*0.22)-r,
                        int(S*0.30)+r, int(S*0.22)+r],
                       fill=(255, 255, 255, max(a, 0)))
        base = Image.alpha_composite(grad, hi)

    # two overlapping bubbles on a transparent layer, then composite
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bsz = int(S * 0.60)
    cyan = (34, 211, 238, 255)    # #22D3EE speaker 1
    pink = (244, 114, 182, 255)   # #F472B6 speaker 2

    b_back = rounded_bubble(bsz, pink, radius=int(bsz*0.22), tail="br")
    b_front = rounded_bubble(bsz, cyan, radius=int(bsz*0.22), tail="bl")

    layer.alpha_composite(b_back, (int(S*0.30), int(S*0.12)))
    # front bubble slightly transparent so the overlap blends to a lighter tint
    fb = b_front.copy()
    fb.putalpha(fb.split()[3].point(lambda a: int(a * 0.90)))
    layer.alpha_composite(fb, (int(S*0.10), int(S*0.30)))

    # transcript lines inside the front bubble (white, rounded)
    d = ImageDraw.Draw(layer)
    lx = int(S*0.10) + int(bsz*0.16)
    ly = int(S*0.30) + int(bsz*0.20)
    lh = int(bsz*0.072)
    gap = int(bsz*0.13)
    widths = [0.58, 0.66, 0.40]
    for i, w in enumerate(widths):
        y = ly + i * gap
        d.rounded_rectangle([lx, y, lx + int(bsz*w), y + lh],
                            radius=lh // 2, fill=(255, 255, 255, 235))

    if bg:
        out = Image.alpha_composite(base, layer)
        return out.convert("RGB")  # iOS: no alpha
    return layer  # Android adaptive foreground (transparent)


build(bg=True).save("tool/voxscribe_icon.png")
build(bg=False).save("tool/voxscribe_icon_fg.png")
print("wrote tool/voxscribe_icon.png and tool/voxscribe_icon_fg.png")
