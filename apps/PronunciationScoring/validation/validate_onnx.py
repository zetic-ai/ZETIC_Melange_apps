"""
validate_onnx.py — behavioral validation of citrinet256_phoneme.onnx
(run from the app folder: python validation/validate_onnx.py)

Checks, on REAL speech, with the EXACT preprocessing the app will use
(pad/truncate to 81760 samples, no normalization, float32 in [-1,1]):
  1. greedy CTC decode produces the right ARPABET phonemes for known utterances
  2. token ids 39-43 (tokenizer specials) never fire; blank=44 dominates silence
  3. GOP path: CTC forced alignment of the TARGET phoneme sequence, per-phoneme
     goodness = mean aligned frame posterior. Correct transcript must score
     clearly higher than a mismatched transcript on the same audio.

This is also the worker's ground-truth harness: the Dart post-processing
pipeline must reproduce the greedy decode and alignment scores produced here.
"""
import json, os, sys
import numpy as np
import soundfile as sf
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "..", "citrinet256_phoneme.onnx")
N_SAMPLES = 81760
BLANK = 44

PHONEMES = ['AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH',
            'IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T','TH',
            'UH','UW','V','W','Y','Z','ZH']  # ids 0..38; 39-43 specials; 44 CTC blank

def preprocess(path):
    """App rule: mono 16 kHz float32, first 81760 samples of the capture.

    IMPORTANT (measured, see model_selection.md): the app must record the FULL
    5.11 s window so the tail is real room tone. NEVER zero-pad — runs of
    digital zeros hit the graph's per-feature normalization (log(2^-24) outlier
    frames) and wreck decode quality (ref1 PER 0.29 -> 0.58 with zero padding).
    For file-based tests this harness approximates room tone with -60 dBFS
    white noise on the padded tail."""
    wav, sr = sf.read(path, dtype="float32")
    assert sr == 16000, f"{path}: expected 16 kHz, got {sr}"
    if wav.ndim > 1: wav = wav.mean(axis=1)
    n = min(len(wav), N_SAMPLES)
    out = np.random.RandomState(7).randn(N_SAMPLES).astype(np.float32) * 1e-3
    out[:n] += wav[:n]
    return out[None, :]

def greedy(logprobs):
    ids = logprobs[0].argmax(-1)
    seq, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK and i < 39:
            seq.append(PHONEMES[i])
        prev = i
    return seq, ids

def ctc_forced_align(logprobs, target_ids):
    """Viterbi over the standard CTC expansion (blank,p1,blank,p2,...,blank).
    Returns per-target-phone list of aligned frames and the alignment path."""
    lp = logprobs[0]                                # [T, C]
    T = lp.shape[0]
    ext = [BLANK]
    for t in target_ids:
        ext += [t, BLANK]
    S = len(ext)
    NEG = -1e30
    dp = np.full((T, S), NEG, dtype=np.float64)
    bp = np.zeros((T, S), dtype=np.int32)
    dp[0, 0] = lp[0, ext[0]]
    if S > 1: dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            best, arg = dp[t-1, s], s
            if s >= 1 and dp[t-1, s-1] > best: best, arg = dp[t-1, s-1], s-1
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s-2] and dp[t-1, s-2] > best:
                best, arg = dp[t-1, s-2], s-2
            if best > NEG/2:
                dp[t, s] = best + lp[t, ext[s]]
                bp[t, s] = arg
    s = S-1 if dp[T-1, S-1] >= dp[T-1, S-2] else S-2
    path = [s]
    for t in range(T-1, 0, -1):
        s = bp[t, s]; path.append(s)
    path.reverse()
    frames = [[] for _ in target_ids]
    for t, s in enumerate(path):
        if ext[s] != BLANK:
            frames[(s-1)//2].append(t)
    return frames

def gop_score(logprobs, target_ids):
    """Per-phoneme GOP: mean posterior of the target phone over its aligned frames."""
    frames = ctc_forced_align(logprobs, target_ids)
    post = np.exp(logprobs[0])                      # [T, C]
    scores = []
    for tid, fr in zip(target_ids, frames):
        scores.append(float(np.mean(post[fr, tid])) if fr else 0.0)
    return scores

def per(ref, hyp):
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=int)
    d[:, 0] = np.arange(len(ref)+1); d[0, :] = np.arange(len(hyp)+1)
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1] + (ref[i-1] != hyp[j-1]))
    return d[-1, -1] / max(len(ref), 1)

TARGETS = {  # CMUdict ARPABET (stress stripped)
  # ls1: real speech (LibriSpeech dev-clean 1272-128104-0001, CC-BY 4.0), 4.8 s
  #      — fills 94% of the window; this is the realistic demo condition.
  "ls1": "N AO R IH Z M IH S T ER K W IH L T ER Z M AE N ER L EH S "
         "IH N T R AH S T IH NG DH AE N HH IH Z M AE T ER".split(),
  # ref1/ref2: macOS `say` synthetic speech, short (49% / 39% window fill)
  #      — deliberate worst-case for window-fill sensitivity.
  "ref1": "DH AH K W IH K B R AW N F AA K S JH AH M P S OW V ER DH AH L EY Z IY D AO G".split(),
  "ref2": "SH IY S EH L Z S IY SH EH L Z B AY DH AH S IY SH AO R".split(),
}

def main():
    audio_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "reference")
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    agg = []
    for clip, target in TARGETS.items():
        p = os.path.join(audio_dir, f"{clip}.wav")
        if not os.path.exists(p):
            print(f"skip {clip} (no wav)"); continue
        x = preprocess(p)
        lp = sess.run(None, {"audio": x})[0]        # [1, 64, 45]
        hyp, ids = greedy(lp)
        specials = int(np.isin(ids, [39, 40, 41, 42, 43]).sum())
        tail_blank = float((ids[-8:] == BLANK).mean())
        e = per(target, hyp)
        tids = [PHONEMES.index(t) for t in target]
        good = gop_score(lp, tids)
        # mismatched transcript control: score the OTHER sentence's phonemes
        other = [k for k in TARGETS if k != clip][0]
        oids = [PHONEMES.index(t) for t in TARGETS[other]]
        bad = gop_score(lp, oids)
        print(f"\n== {clip}")
        print("   greedy :", " ".join(hyp))
        print("   target :", " ".join(target))
        print(f"   PER={e:.3f}  specials_fired={specials}  tail_blank_frac={tail_blank:.2f}")
        print(f"   GOP correct-text  mean={np.mean(good):.3f}  min={np.min(good):.3f}")
        print(f"   GOP mismatched    mean={np.mean(bad):.3f}")
        agg.append((clip, e, float(np.mean(good)), float(np.mean(bad)), specials))
    print("\nSUMMARY (clip, PER, gop_correct, gop_mismatch, specials):")
    for row in agg: print("  ", row)

if __name__ == "__main__":
    main()
