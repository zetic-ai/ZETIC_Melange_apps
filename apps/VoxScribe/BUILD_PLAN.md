# BUILD_PLAN — VoxScribe (GATE-2 deliverable)

Worker agent, branch `app/voxscribe`, worktree `/Users/ajayshah/Desktop/ZETIC/voxscribe-wt`.
App code target: `apps/VoxScribe/Flutter/`. **No Dart written yet — this is plan-only, STOP after.**

Built strictly to `apps/VoxScribe/SPEC.md`. SDK realities honored per `agentic-workflow-docs/CLAUDE.md` §5.

---

## 0. Verified facts (checked, not assumed)

- **`zetic_mlange` = 1.8.1** — confirmed from PyroGuard's `.flutter-plugins-dependencies` and
  `package_config.json` (`zetic_mlange-1.8.1`), and matches PyroGuard HANDOFF (`zetic_mlange 1.8.1`).
  Will pin exactly `1.8.1`.
- **SDK API surface** (read from `~/.pub-cache/.../zetic_mlange-1.8.1/lib/src/`):
  - `static Future<ZeticMLangeModel> create({required String personalKey, required String name, int? version, ModelMode modelMode = ModelMode.runAuto})` — async factory (matches CLAUDE.md §5).
  - `List<Tensor> run([List<Tensor> inputs])` — **synchronous**, returns output tensors.
  - `void close()`.
  - `Tensor.float32List(Float32List, {List<int>? shape})`, **`Tensor.int32List(Int32List, {shape})`** (int32 IS supported — decoder needs it), `outputs.first.asFloat32List()`, `asInt32List()`.
  - `DataType.int32` = nativeValue 11, 4 bytes/elem.
- **Log-mel is NOT in the Flutter plugin.** The native references do mel + detokenization via the SDK-internal
  `WhisperWrapper` (`com.zeticai.mlange.feature.automaticspeechrecognition...` / Swift `WhisperWrapper`). The Flutter
  `zetic_mlange` package exports only `model`, `tensor`, `types`, `hf_model`, `llm_model` — **no ASR/whisper feature
  helper**. ⇒ **log-mel and detokenization MUST be ported to pure Dart.** (See decisions 2 & 3.)
- **`vocab.json`** (from `apps/whisper-tiny`, 835 KB, 50258 entries, token→id, GPT-2 byte-level e.g. `Ġthe`→264,
  `<|endoftext|>`→50257) will be copied and bundled as a Flutter asset for pure-Dart detokenization.
- **Greedy decode loop** (from Android `WhisperDecoder.kt`, iOS mirror): ids `IntArray(448){50256}`, `ids[0]=SOT 50258`;
  mask `IntArray(448){0}`, `mask[0]=1`; `idx=1`; per step run decoder, `vocab=51865`, read logit row
  `[(idx-1)*51865 .. idx*51865)`, argmax, stop on EOT 50257, else write `ids[idx]`/`mask[idx]`, `idx++`. Positional
  input order to `run`: `(ids, enc_hidden, enc_mask)`.
- **AudioSampler.kt**: 16 kHz, mono, `ENCODING_PCM_FLOAT` — confirms target format.

---

## 1. Build plan

### 1.1 File tree (under `Flutter/`, mapped onto CLAUDE.md §4, adapted for audio)

```
Flutter/
  pubspec.yaml                 # zetic_mlange: 1.8.1; assets: demo clip, vocab.json, mel_filters
  assets/
    demo_2spk.wav              # FLOOR input: bundled <=10s 2-speaker 16k mono clip  (decision 1)
    vocab.json                 # copied from apps/whisper-tiny  (detokenization)
    mel_filters_80.bin         # openai-whisper 80x201 mel filterbank, float32  (decision 2)
  lib/
    main.dart
    screens/
      loading_screen.dart      # 3-model download + warm-up (dummy inference each), progress bar
      main_screen.dart         # the demo: load clip -> run pipeline -> live transcript + timeline + HUD
    services/
      melange_service.dart     # owns the 3 model handles; create/warmup/run/close; lives on ONE isolate
      pipeline_isolate.dart    # long-lived worker isolate host: owns MelangeService, streams progress
      preprocessor.dart        # decode wav -> downmix(ch0) -> resample 16k -> [-1,1];
                               #   segmentation 10s window [1,1,160000]; per-span 30s pad -> log-mel [1,80,3000]
      log_mel.dart             # pure-Dart STFT(n_fft=400,hop=160,Hann)+melfilter(80)+log10+clamp/scale
      postprocessor.dart       # powerset decode (7-class table); onset/offset segmentation; greedy decode argmax
      diarization_fusion.dart  # for each SpeakerSegment span -> encoder+decoder -> TranscriptLine tagged w/ speaker
      detokenizer.dart         # pure-Dart GPT-2 byte-level BPE: id->token->byte-decode (vocab.json)
    models/
      speaker_segment.dart     # {double start, end; int speaker}
      transcript_line.dart     # {int speaker; double start, end; String text}
      stage_timings.dart       # per-stage ms + RTF for the HUD
    widgets/
      transcript_view.dart     # scrolling speaker-colored lines
      timeline_widget.dart     # who-spoke-when speaker bands over time (CustomPaint)
      hud.dart                 # "On-device . No cloud" badge + per-stage latency/RTF + served-artifact line
  test/
    resample_test.dart
    downmix_test.dart
    waveform_norm_test.dart
    segmentation_window_test.dart
    frame_time_map_test.dart
    powerset_decode_test.dart
    logsoftmax_test.dart
    onset_offset_segment_test.dart
    log_mel_test.dart
    whisper_span_pad_test.dart
    greedy_decode_test.dart
    token_dtype_test.dart
    detokenizer_test.dart
    fusion_attribution_test.dart
    benchmark/
      hot_path_benchmark.dart
```

### 1.2 Pipeline orchestration order (the FLOOR)

1. Load clip from asset → decode PCM → **downmix stereo→mono (ch0)** → **resample to 16 kHz** → float32 `[-1,1]` (int16/32768 if needed). (once per clip)
2. **Segmentation**: take/zero-pad to exactly 160000 samples → `Tensor.float32List(shape:[1,1,160000])` → `run` → `[1,589,7]` log-softmax.
3. **Powerset decode**: per frame argmax(7) → decode-table → `labels[589][3]`; onset/offset state machine (onset=offset=0.5, on logsoftmax compared after argmax — argmax is exp-invariant); segment time `frame*0.016875 + 0.0309688`; merge same-speaker gaps ≤0.50; drop <0.30 → `List<SpeakerSegment>`.
4. **Fusion (diarize-THEN-transcribe), per segment span**:
   a. slice span samples → zero-pad/truncate to 480000 (30 s) → **log-mel `[1,80,3000]`**.
   b. **encoder** `run([logmel])` → `[1,1500,384]` (reused as decoder `enc_hidden` unchanged).
   c. **decoder greedy loop** (≤448 steps): build int32 ids/mask, `run([ids, enc_hidden, enc_mask])`, read row idx-1, argmax, EOT-stop.
   d. **detokenize** ids (skip specials) → text → emit `TranscriptLine{speaker, start, end, text}`.
   e. **stream this line to the UI immediately** (progressive render — R4).
5. UI paints transcript + timeline; HUD shows per-stage ms + RTF + on-device badge.

### 1.3 Threading / isolate model — **one long-lived dedicated isolate owning all 3 handles**

Decision: spawn **one** background isolate at app start; it creates and warms all three `ZeticMLangeModel`
handles and runs the **entire** pipeline (segmentation + the N× encoder/decoder loops). It posts progress
messages (segments found, each finished `TranscriptLine`, per-stage timings) back to the UI isolate via `SendPort`.

Justification (ties to VALIDATION.md Tier B "Isolate and copy cost"):
- **The SDK binds a model handle to one isolate** — the native pointer is wrapped in a non-sendable Dart object,
  and `run()` is **synchronous**. So a handle created on isolate A cannot be `run` on isolate B. All `create` + `run`
  + `close` for a given model must occur on the **same** isolate. ⇒ rules out PyroGuard's per-call `compute()` pattern.
- The pipeline is **multi-second** (3 model loads + log-mel + N×up-to-448 synchronous decode steps). Running it inline
  on the UI isolate would freeze the demo's scroll/animation. ⇒ rules out inline.
- ⇒ **long-lived worker isolate** is the only correct option: create-once, warm-once, keep handles for app
  lifetime, stream results out. Cross-isolate copy is tiny (a `List<SpeakerSegment>` and short
  `TranscriptLine` strings) — the heavy `[1,1500,384]` encoder output (576k floats) and `[1,448,51865]` decoder
  logits **never cross the boundary**; they stay inside the worker isolate and are reused across decode steps.
- Warm-up: one dummy inference per model right after load (Tier B "Model lifecycle"), on the worker isolate.

---

## 2. Exact Tier A test list

Every test uses hand-built input + known expected output. Maps 1:1 to SPEC "Validation focus".

1. **resample_test** (sample-rate trap) — 48000-sample linear ramp @48 kHz → resampler → assert length == 16000
   (N input sec ⇒ N output sec), endpoints preserved within tol; 16000 @16 kHz → identity (no-op).
2. **downmix_test** (mono downmix) — interleaved stereo `[L0,R0,L1,R1,...]` with distinct ch0 → assert mono == ch0
   samples (SPEC: reference uses ch0), length halved.
3. **waveform_norm_test** (normalization) — int16 `[32767,0,-32768]` → assert `/32768` only → `[~+1.0, 0.0, -1.0]`,
   all in `[-1,1]`, and float input passes through unchanged (no double-normalize).
4. **segmentation_window_test** (window framing) — (a) 80000 samples → zero-padded tail to exactly **160000**, shape
   `[1,1,160000]`, pad region == 0; (b) 200000 samples → truncated to 160000.
5. **frame_time_map_test** (frame→time map) — assert frame count constant **589**; frame 0 → 0.0309688 s,
   frame 588 → `588*0.016875+0.0309688`, frame 100 → exact value; scale == 270/16000.
6. **powerset_decode_test** (7-class powerset, incl. overlaps) — build a `[1,589,7]` tensor whose per-frame argmax
   hits **each** class 0..6; assert decode table: 0→{}, 1→{0}, 2→{1}, 3→{2}, **4→{0,1}, 5→{0,2}, 6→{1,2}**;
   assert binary `labels[frame][localSpk]` matrix (7 classes ≠ 7 speakers).
7. **logsoftmax_test** (log-softmax) — known log-softmax row → assert `exp(row).sum ≈ 1.0`; assert argmax(row) ==
   argmax(exp(row)) (invariant); assert any prob threshold compares against `exp(value)`, not the raw log value.
8. **onset_offset_segment_test** (segmentation state machine + fusion-precursor) — hand-built per-frame binary
   activity for one local speaker producing two runs separated by a >0.50 s gap, plus one sub-0.30 s blip →
   assert: blip dropped (min_duration_on), runs within ≤0.50 s gap merged (min_duration_off), 2 final
   `SpeakerSegment{start,end,speaker}` with times via the frame map.
9. **log_mel_test** (Whisper log-mel exactness) — known waveform (e.g. 440 Hz sine, 16 kHz) → assert output shape
   `[1,80,**3000**]`, n_fft=400/hop=160 frame count == 3000, and clamp/scale formula
   `log_spec=max(log_spec, max-8.0); (log_spec+4)/4` matches a **bundled golden reference vector**
   (computed offline from openai/whisper) within tol.
10. **whisper_span_pad_test** (30 s span framing) — (a) 32000-sample (2 s) span → zero-pad tail to **480000**
    (30 s) before mel; (b) 600000-sample span → truncate to 480000; assert resulting mel is `[1,80,3000]`.
11. **greedy_decode_test** (termination + idx-1 indexing + seeding) — inject a fake decoder whose returned
    `[1,448,51865]` logits make row `idx-1` argmax to a scripted sequence `[t1, t2, EOT]`; assert: initial
    `ids[0]==50258(SOT)`, rest `50256(pad)`, `mask[0]==1` rest 0, `idx` starts at 1; loop reads row
    **`(idx-1)*51865 .. idx*51865`** (NOT row 0); stops on EOT 50257; never writes past `idx`; buffer length
    stays 448; collected == `[t1, t2]`.
12. **token_dtype_test** (int32 dtype / byte length) — build ids & mask Tensors → assert `DataType.int32`,
    448 elements, **`lengthInBytes == 448*4 == 1792`** each; assert built via `Tensor.int32List`.
13. **detokenizer_test** (byte-level BPE detok) — known id sequence (e.g. ids for `Ġthe`/`Ġworld` from vocab.json)
    → assert exact decoded string with `Ġ`→space (byte-level reverse), specials (id ≥ 50257, SOT 50258) skipped
    and never emitted.
14. **fusion_attribution_test** (fusion attribution) — hand-built 2 `SpeakerSegment`
    (`s0:[0.0,2.0]`, `s1:[2.0,4.0]`) + a stubbed transcriber returning `"hello"`/`"world"` → assert exactly two
    `TranscriptLine` `{speaker:1,"hello",0..2}`, `{speaker:2,"world",2..4}`, in timeline order — text tagged with
    its segment's speaker **by construction**.

### A4 hot-path micro-benchmark (`benchmark/hot_path_benchmark.dart`)

Feed **mock tensors of the real shapes** through the full pure-Dart hot path; mock the 3 `run()` calls with
fixed-shape tensors so only Dart cost is measured. Report median over many iterations. Stages timed:
- **log-mel**: 480000-sample span → `[1,80,3000]` (the heaviest Dart stage: STFT + 80-mel matmul).
- **powerset decode + onset/offset segmentation** over a mock `[1,589,7]`.
- **greedy decode argmax**: argmax over **51865** logits × ~50 steps (realistic span), reading row idx-1 each step
  from a mock `[1,448,51865]` buffer.
- **detokenization** of ~50 ids.
Record the median → becomes the Tier B baseline (post-processing budget, not end-to-end device latency).

---

## 3. Spec ambiguities / decisions needed (GATE-2 — my one chance to ask)

1. **Demo audio clip (FLOOR input).** No clip exists in the repo. The SPEC floor requires a bundled **≤10 s,
   2-speaker, 16 kHz mono** clip. **Decision needed:** should I (a) source/synthesize one (e.g. two TTS voices or
   a CC-licensed 2-speaker snippet, trimmed to ≤10 s, transcoded to 16 kHz mono WAV) and bundle it, or (b) will you
   provide the clip? A known-script clip also lets me hand-write the expected transcript for a visual sanity check.
2. **Log-mel = pure Dart.** Confirmed: the Flutter `zetic_mlange` plugin does **not** expose the native
   `WhisperWrapper` mel (only `model`/`tensor`/`types`). I will port log-mel to pure Dart
   (STFT n_fft=400, hop=160, Hann; 80-mel; log10; clamp `max-8`, scale `(x+4)/4`) and bundle the **exact
   openai/whisper 80×201 mel filterbank** as an asset (`mel_filters_80.bin`) so it matches the reference bit-for-bit.
   **Confirm** this approach and that I may generate the bundled filterbank + the A9 golden vector from the
   `openai-whisper` package offline. (Alternative: a native MethodChannel wrapping the platform WhisperWrapper —
   more work, platform-specific, not recommended for the FLOOR.)
3. **Detokenization = pure Dart.** Same reason — `decodeToken` isn't in the Flutter plugin. I will port GPT-2
   byte-level BPE decode (id→token via bundled `vocab.json`, then byte-decoder reverse of `bytes_to_unicode`).
   `vocab.json` has 50258 entries (0..50257) but decoder logits are 51865 wide; ids ≥ 50258 are special/timestamp
   tokens with no vocab entry. **Confirm:** skip every id ≥ 50257 and ≤ ... i.e. emit only non-special ids
   (drop SOT 50258, EOT 50257, pad 50256, and any timestamp/special ≥ 50258). With `<|notimestamps|>`-style SOT-only
   seeding they shouldn't appear, but I'll guard.
4. **`zetic_mlange` pin = `1.8.1`** (matches PyroGuard). Confirm OK to pin exactly (vs `^1.8.1`).
5. **Personal key.** The key is embedded in the client (CLAUDE.md / Tier C). **Decision needed:** reuse PyroGuard's
   ZETIC personal key — where do I read it from (PyroGuard's `lib/` source isn't in this checkout; only build
   artifacts are present)? Please paste the key or point me to it.
6. **iOS signing team + bundle id.** PyroGuard used team **WVJ22PPYBP**. Confirm same team and the desired bundle id
   (proposing `ai.zetic.voxscribe`). Android applicationId likewise (proposing `ai.zetic.voxscribe`).
7. **Mono downmix method.** SPEC says "ch0 or average; reference uses ch0." I'll use **ch0** to match the reference —
   confirm.
8. **Decoder `enc_mask` semantics.** Positional order is `(ids, enc_hidden, enc_mask)`. In the native reference the
   3rd tensor is the **448-long decoder attention mask** (1 for filled positions). I'll follow the native positional
   behavior (enc_mask = that 448 int32 decoder mask), not an encoder-frame mask. Confirm.
9. **modelMode = RUN_AUTO** for all three (per SPEC). Confirm (no client mode steers backend anyway — CLAUDE.md §5).
10. **"On-device / airplane-mode" signal.** Proposing a **static "On-device · No cloud" badge** + per-stage
    latency/RTF readout on the HUD (no network plugin, no live airplane-mode detection — the app simply never makes a
    network call after the one-time model download). Confirm a static badge + RTF HUD satisfies the must-have, or do
    you want an actual connectivity probe?
11. **Fusion = diarize-then-transcribe (FLOOR).** SPEC names this the GATE-2 confirm point. I will build the floor
    fusion and **not** pursue the "transcribe-once-then-assign-dominant-speaker" alternative (needs Whisper timestamp
    tokens). Confirm.
12. **30 s span padding.** Encoder input is fixed `[1,80,3000]`, so every span (even 2 s) is zero-padded to 480000
    samples → 3000 frames. This is mandatory by shape but Whisper can hallucinate on long trailing silence. I'll
    accept it for the FLOOR (and may add a no-speech/`avg_logprob`-style guard as stretch). Confirm acceptable.

---

## 4. Floor-first commitment

I will build the **SPEC FLOOR first** and treat everything else as stretch, so there is always a shippable
screen-recordable demo before Monday KST:

- **FLOOR (guaranteed video):** file-based **≤10 s** bundled clip · **single** 10 s segmentation window
  (no stitching, no clustering — local speaker slots ARE global) · **diarize-then-transcribe** · 2 speakers ·
  **static** who-spoke-when timeline · on-device badge + latency/RTF HUD.
- **Stretch 1:** sliding-window segmentation for longer audio.
- **Stretch 2:** live mic instead of file.
- **Stretch 3:** full diarization (4th embedding model + pure-Dart clustering) — needs a separate GATE-0 upload.

Stretch work begins only after the FLOOR passes Tier A + satisfies Tier B and a clean demo video is achievable.

---

**STOP — awaiting GATE-2 human approval before writing any Dart / running `flutter create`.**
