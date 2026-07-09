"""export_golden.py — export golden fixtures for the Dart post-processing tests.

Runs citrinet256_phoneme.onnx on the committed reference wavs (the EXACT app
preprocessing) and writes, per clip, the raw logprobs [64][45] plus the
ground-truth outputs of the Python reference pipeline (validate_onnx.py):
greedy decode, CTC forced-alignment frame sets, per-phoneme GOP, and the
blank-frame fraction (window-fill proxy).

The Dart golden_parity_test loads the logprobs from each fixture, re-runs the
Dart aligner/scorer, and asserts it reproduces greedy / aligned frames / GOP
within 1e-3 (NOT bit-exact: on-device the served artifact differs in precision).

Run from the app folder:  python validation/export_golden.py
Writes: Flutter/test/fixtures/golden_<clip>.json
"""
import json, os
import numpy as np
import onnxruntime as ort

# Reuse the reference pipeline verbatim so the fixtures ARE its output.
from validate_onnx import (
    PHONEMES, BLANK, N_SAMPLES,
    preprocess, greedy, ctc_forced_align, gop_score, TARGETS,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "..", "citrinet256_phoneme.onnx")
REF = os.path.join(HERE, "reference")
OUT = os.path.join(HERE, "..", "Flutter", "test", "fixtures")


def main():
    os.makedirs(OUT, exist_ok=True)
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    index = []
    for clip, target in TARGETS.items():
        p = os.path.join(REF, f"{clip}.wav")
        if not os.path.exists(p):
            print(f"skip {clip} (no wav)")
            continue
        x = preprocess(p)
        lp = sess.run(None, {"audio": x})[0]        # [1, 64, 45]
        T, C = lp.shape[1], lp.shape[2]
        tids = [PHONEMES.index(t) for t in target]
        hyp, ids = greedy(lp)
        frames = ctc_forced_align(lp, tids)
        gop = gop_score(lp, tids)
        blank_frac = float((ids == BLANK).mean())
        fixture = {
            "clip": clip,
            "shape": [T, C],
            "target_ids": tids,
            "target_phonemes": target,
            "logprobs": [float(v) for v in lp[0].reshape(-1)],  # row-major T*C
            "greedy": hyp,
            "aligned_frames": [list(map(int, f)) for f in frames],
            "gop": [float(g) for g in gop],
            "blank_fraction": blank_frac,
        }
        out = os.path.join(OUT, f"golden_{clip}.json")
        with open(out, "w") as f:
            json.dump(fixture, f)
        index.append(clip)
        print(f"{clip}: greedy_len={len(hyp)} gop_mean={np.mean(gop):.3f} "
              f"blank_frac={blank_frac:.3f} -> {os.path.relpath(out, HERE)}")
    with open(os.path.join(OUT, "golden_index.json"), "w") as f:
        json.dump(index, f)
    print("wrote", len(index), "fixtures +", "golden_index.json")


if __name__ == "__main__":
    main()
