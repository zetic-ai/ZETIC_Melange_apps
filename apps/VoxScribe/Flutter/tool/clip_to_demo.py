"""Turn any audio clip into VoxScribe demo data (segments + transcript).

Runs the REAL pyannote segmentation ONNX offline (onnxruntime) on a clip,
mirrors the app's powerset + onset/offset post-processing, remaps local speaker
slots to 1-based labels by first appearance, and prints the Dart literals to
paste into pipeline_isolate.dart (kDemoReferenceSegments + kDemoTranscript).

This is the "swap in your own clip" path while live on-device transcription is
disabled (the no-cache decoder OOMs). Segments are the model's real output;
transcript text you provide (one line per speaker turn, in time order).

Usage:
  # 1) convert your recording to 16 kHz mono WAV (any format in):
  #    afconvert -f WAVE -d LEI16@16000 -c 1 input.m4a demo.wav
  # 2) run this, passing the transcript lines in spoken order:
  python tool/clip_to_demo.py demo.wav \
      "First speaker's words" "Second speaker's words" "Third ..." ...
  # 3) paste the printed Dart into lib/services/pipeline_isolate.dart,
  #    copy demo.wav -> assets/demo_2spk.wav, flutter run.
"""
import sys, wave, numpy as np, onnxruntime as ort

ONNX = "/Users/ajayshah/Desktop/ZETIC/voxscribe-wt/apps/VoxScribe/diarization/pyannote_segmentation_static.onnx"
SR = 16000
WIN = 160000  # 10 s window (single-window floor: <=3 speakers, no clustering)
SCALE, OFFSET = 270 / SR, 991 / SR * 0.5
ONSET = OFFSET_TH = 0.5
MIN_ON, MIN_OFF = 0.30, 0.50
POW = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2]]


def load_mono16k(path):
    w = wave.open(path, "rb")
    n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
    pcm = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
    w.close()
    if ch > 1:
        pcm = pcm[0::ch]
    if sr != SR:
        raise SystemExit(f"Resample to 16 kHz first (got {sr}). Use afconvert.")
    return pcm


def segments(pcm):
    win = np.zeros(WIN, np.float32)
    win[: min(WIN, pcm.size)] = pcm[:WIN]
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    y = sess.run(None, {sess.get_inputs()[0].name: win.reshape(1, 1, WIN)})[0][0]
    arg = y.argmax(1)
    labels = np.zeros((y.shape[0], 3), bool)
    for f, c in enumerate(arg):
        for s in POW[c]:
            labels[f][s] = True
    raw = []  # (start_s, end_s, local_slot)
    for s in range(3):
        runs, st = [], None
        for f, a in enumerate(labels[:, s]):
            if a and st is None:
                st = f
            elif not a and st is not None:
                runs.append((st, f)); st = None
        if st is not None:
            runs.append((st, len(labels)))
        times = [[r[0] * SCALE + OFFSET, r[1] * SCALE + OFFSET] for r in runs]
        merged = []
        for t in times:
            if merged and t[0] - merged[-1][1] <= MIN_OFF:
                merged[-1][1] = t[1]
            else:
                merged.append(t)
        for a, b in merged:
            if b - a >= MIN_ON:
                raw.append((a, b, s))
    raw.sort(key=lambda r: r[0])
    # remap local slots -> 0-based label by first appearance (clean Speaker 1/2/3)
    remap, nxt = {}, 0
    out = []
    for a, b, slot in raw:
        if slot not in remap:
            remap[slot] = nxt; nxt += 1
        out.append((round(a, 2), round(b, 2), remap[slot]))
    return out


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path, texts = sys.argv[1], sys.argv[2:]
    segs = segments(load_mono16k(path))
    print(f"// {len(segs)} segments, {max((s for *_, s in segs), default=-1)+1} speakers\n")
    print("final List<SpeakerSegment> kDemoReferenceSegments = <SpeakerSegment>[")
    for a, b, s in segs:
        print(f"  const SpeakerSegment(start: {a}, end: {b}, speaker: {s}),")
    print("];\n")
    print("const List<String> kDemoTranscript = <String>[")
    for i in range(len(segs)):
        line = texts[i] if i < len(texts) else "..."
        print(f"  {line!r},".replace("\\'", "'") if "'" not in line else f'  "{line}",')
    print("];")
    if len(texts) != len(segs):
        print(f"\n// NOTE: {len(segs)} segments but {len(texts)} transcript lines given.")


if __name__ == "__main__":
    main()
