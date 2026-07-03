"""gen_sentences.py — build the bundled practice-sentence asset OFFLINE.

For each curated sentence: tokenize into words, look each word up in CMUdict
(first pronunciation, stress digits stripped), map ARPABET labels to the model's
class ids (0..38, labels.txt order), and record a per-word [start, end) span into
the flat phoneme-id list. A read-time estimate (phoneme_count * SEC_PER_PHONEME)
is stored so the app can keep sentences in the 3.5–5.0 s window-fill band.

There is NO runtime G2P — the app reads this asset directly.

Run:  python tools/gen_sentences.py
Writes: assets/sentences.json
Requires: pip install cmudict
"""
import json, os, re, sys
import cmudict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "sentences.json")

# Must match Flutter lib/models/phonemes.dart (ids 0..38) and app-root labels.txt.
ARPABET = ['AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G',
           'HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T',
           'TH','UH','UW','V','W','Y','Z','ZH']
ID = {p: i for i, p in enumerate(ARPABET)}

# Must match kSecondsPerPhoneme in lib/models/sentence.dart / the asset test.
SEC_PER_PHONEME = 0.105

# Curated general-English sentences (a couple carry a light on-device-AI /
# industrial flavor). Sized so phoneme_count lands in [33, 47] -> 3.5–5.0 s.
SENTENCES = [
    "The quick brown fox quietly jumps over the lazy sleeping dog.",
    "Please call Stella and ask her to bring these things along.",
    "She carefully sells bright seashells beside the sunny seashore.",
    "Reading a short passage aloud each day improves your speech.",
    "Our engineers ran the whole model directly on the phone.",
    "The sensors measured every reading across the factory floor.",
    "Thank you so much for helping me practice speaking clearly.",
    "A gentle rain fell softly over the quiet mountain village.",
]

# Fallback pronunciations for words CMUdict may lack (stress-stripped ARPABET).
OVERRIDES = {}


def phones_for_word(word, cmu):
    key = re.sub(r"[^a-z']", "", word.lower())
    if not key:
        return None, []
    if key in OVERRIDES:
        arps = OVERRIDES[key]
    else:
        prons = cmu.get(key)
        if not prons:
            return key, None
        arps = [re.sub(r"\d", "", p) for p in prons[0]]
    ids = []
    for a in arps:
        if a not in ID:
            return key, None
        ids.append(ID[a])
    return key, ids


def main():
    cmu = cmudict.dict()
    out = []
    ok = True
    for text in SENTENCES:
        words = text.replace("’", "'").split()
        clean_words, phoneme_ids, spans = [], [], []
        bad = False
        for w in words:
            key, ids = phones_for_word(w, cmu)
            if ids is None:
                print(f"  !! no CMUdict entry for '{key}' in: {text}")
                bad = True
                ok = False
                break
            start = len(phoneme_ids)
            phoneme_ids.extend(ids)
            spans.append([start, len(phoneme_ids)])
            clean_words.append(re.sub(r"[.,!?;:]", "", w))
        if bad:
            continue
        est = round(len(phoneme_ids) * SEC_PER_PHONEME, 3)
        flag = "" if 3.5 <= est <= 5.0 else "  <-- OUT OF 3.5-5.0s BAND"
        print(f"  {len(phoneme_ids):2d} phones  est={est:.2f}s{flag}  {text}")
        out.append({
            "text": text,
            "words": clean_words,
            "phoneme_ids": phoneme_ids,
            "spans": spans,
            "est_seconds": est,
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"sentences": out}, f, indent=2)
    print(f"\nwrote {len(out)} sentences -> {os.path.relpath(OUT, HERE)}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
