"""
export.py — PronunciationScoring (speech / ASR-encoder family recipe, first of its family)

Exports Peacockery/citrinet-256-phoneme-en (NeMo EncDecCTCModelBPE, 9.7M params,
ARPABET phoneme CTC) to a SINGLE static-shape ONNX that takes RAW WAVEFORM:

    input : audio  float32[1, 81760]   (5.11 s @ 16 kHz mono, range [-1, 1])
    output: logprobs float32[1, 64, 45] (64 CTC frames x 45 classes, log-softmax)

Why raw waveform: the NeMo model normally needs an 80-bin log-mel frontend.
Rebuilding that DSP in Dart is a large, error-prone surface. Instead this recipe
bakes the EXACT NeMo AudioToMelSpectrogramPreprocessor math into the ONNX graph
using only standard ops (Pad/Conv/MatMul/Log/ReduceMean/Slice), verified
numerically against the real NeMo preprocessor. The app's preprocessing is then
just: mic PCM16 -> float32/32768, pad/truncate to 81760 samples.

Why 81760 samples: torch.stft(center=True) gives 1 + 81760/160 = 512 mel frames,
a multiple of the model's pad_to=16 (so NeMo would add no padding), and the
encoder's 3 stride-2 blocks give exactly 512/8 = 64 output frames. NeMo counts
floor(81760/160) = 511 frames as "valid"; we replicate that too (normalization
over 511 frames, last frame zeroed, encoder length = 511).

Opset: 12 — verified working (the graph is conv/matmul-only, no transformer
ops; 13 and 14 also export cleanly, 12 chosen to match the PyroGuard
known-good-with-Melange precedent).
Re-run:  python export.py            (needs: nemo_toolkit[asr], torch, onnx,
                                      onnxruntime, librosa, huggingface_hub)
"""

import numpy as np
import torch
import torch.nn as nn

SAMPLE_RATE = 16000
N_SAMPLES = 81760          # 5.11 s
N_FFT = 512
WIN_LENGTH = 400           # 25 ms
HOP = 160                  # 10 ms
N_MELS = 80
N_FRAMES = 1 + N_SAMPLES // HOP        # 512 stft frames (center=True)
VALID_FRAMES = N_SAMPLES // HOP        # 511 "valid" frames per NeMo get_seq_len
PREEMPH = 0.97
LOG_GUARD = 2.0 ** -24
NORM_CONST = 1e-5          # NeMo CONSTANT added to std
OPSET = 12
OUT_ONNX = "citrinet256_phoneme.onnx"


class MelFrontend(nn.Module):
    """Bit-exact replica of NeMo FilterbankFeatures (eval mode) for a fixed-length
    input, expressed in ONNX-friendly standard ops.

    NeMo config: dither(off at eval), preemph 0.97, hann(400, periodic=False)
    zero-padded to n_fft=512, torch.stft(center=True, pad_mode='constant'),
    power-2 magnitude, librosa slaney mel (fmin 0, fmax 8000), log(x + 2^-24),
    per-feature normalization over the 511 valid frames (unbiased std + 1e-5),
    frames beyond valid zeroed.
    """

    def __init__(self):
        super().__init__()
        import librosa

        # window: hann(400, periodic=False) centered in 512 (torch.stft behavior)
        win = torch.hann_window(WIN_LENGTH, periodic=False)
        pad = (N_FFT - WIN_LENGTH) // 2
        win512 = torch.zeros(N_FFT)
        win512[pad:pad + WIN_LENGTH] = win

        # DFT basis as a conv kernel: out channels = 257 real + 257 imag
        n = torch.arange(N_FFT, dtype=torch.float64)
        k = torch.arange(N_FFT // 2 + 1, dtype=torch.float64).unsqueeze(1)
        ang = 2.0 * torch.pi * k * n / N_FFT
        cos_b = (torch.cos(ang) * win512.double()).float()   # [257, 512]
        sin_b = (-torch.sin(ang) * win512.double()).float()  # [257, 512]
        dft = torch.cat([cos_b, sin_b], dim=0).unsqueeze(1)  # [514, 1, 512]
        self.register_buffer("dft", dft)

        mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                                  fmin=0.0, fmax=SAMPLE_RATE / 2.0)  # slaney norm
        self.register_buffer("mel_fb", torch.tensor(mel, dtype=torch.float32))

    def forward(self, audio):                      # audio: [1, 81760]
        # pre-emphasis, first sample passed through (NeMo semantics)
        x = torch.cat((audio[:, :1], audio[:, 1:] - PREEMPH * audio[:, :-1]), dim=1)
        # center pad 256 zeros each side (torch.stft center=True, pad_mode='constant')
        x = torch.nn.functional.pad(x.unsqueeze(1), (N_FFT // 2, N_FFT // 2))  # [1,1,82272]
        # framed windowed DFT as one conv
        spec = torch.nn.functional.conv1d(x, self.dft, stride=HOP)  # [1, 514, 512]
        re, im = spec[:, :N_FFT // 2 + 1], spec[:, N_FFT // 2 + 1:]
        power = re * re + im * im                                   # [1, 257, 512]
        mel = torch.matmul(self.mel_fb, power)                      # [1, 80, 512]
        logmel = torch.log(mel + LOG_GUARD)
        # per-feature normalization over the 511 valid frames
        valid = logmel[:, :, :VALID_FRAMES]
        mean = valid.mean(dim=2, keepdim=True)
        var = ((valid - mean) ** 2).sum(dim=2, keepdim=True) / (VALID_FRAMES - 1)
        std = torch.sqrt(var) + NORM_CONST
        norm = (logmel - mean) / std
        # zero everything beyond the valid frames (frame 511)
        return torch.cat(
            (norm[:, :, :VALID_FRAMES], torch.zeros_like(norm[:, :, VALID_FRAMES:])),
            dim=2,
        )                                                            # [1, 80, 512]


class CitrinetPhonemeE2E(nn.Module):
    """MelFrontend + NeMo Citrinet-256 encoder + CTC decoder, fixed input length.

    Length is a constant (511 mel frames); tracing constant-folds every mask, so
    masked convs / squeeze-excite keep exact NeMo eval semantics in a fully
    static graph.
    """

    def __init__(self, nemo_model):
        super().__init__()
        self.frontend = MelFrontend()
        self.encoder = nemo_model.encoder
        self.decoder = nemo_model.decoder
        self.register_buffer("length", torch.tensor([VALID_FRAMES], dtype=torch.int64))

    def forward(self, audio):                       # [1, 81760]
        mel = self.frontend(audio)                  # [1, 80, 512]
        enc, _ = self.encoder(audio_signal=mel, length=self.length)  # [1, 640, 64]
        return self.decoder(encoder_output=enc)     # [1, 64, 45] log-probs


def main():
    from huggingface_hub import hf_hub_download
    import nemo.collections.asr as nemo_asr

    nemo_path = hf_hub_download("Peacockery/citrinet-256-phoneme-en", "model.nemo")
    m = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
    m.eval()

    wrapper = CitrinetPhonemeE2E(m)
    wrapper.eval()

    # ---- parity check vs the real NeMo pipeline before export ----
    torch.manual_seed(0)
    wav = torch.randn(1, N_SAMPLES) * 0.05
    with torch.no_grad():
        ref_mel, _ = m.preprocessor(
            input_signal=wav, length=torch.tensor([N_SAMPLES], dtype=torch.int64))
        got_mel = wrapper.frontend(wav)
        mel_err = (ref_mel - got_mel).abs().max().item()
        ref_lp, _, _ = m.forward(
            input_signal=wav,
            input_signal_length=torch.tensor([N_SAMPLES], dtype=torch.int64))
        got_lp = wrapper(wav)
        lp_err = (ref_lp - got_lp).abs().max().item()
    print(f"frontend max|diff| vs NeMo preprocessor: {mel_err:.3e}")
    print(f"full-graph max|diff| vs NeMo forward:    {lp_err:.3e}")
    assert mel_err < 1e-3 and lp_err < 1e-3, "parity with NeMo lost — do not export"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (torch.randn(1, N_SAMPLES),),
            OUT_ONNX,
            input_names=["audio"],
            output_names=["logprobs"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamic_axes=None,          # static shapes or bust
            dynamo=False,               # legacy tracing exporter
        )

    # ---- graph cleanup: make the constant-length machinery ACTUALLY constant ----
    # The traced graph carries Shape/Expand/Where mask chains from NeMo's
    # MaskedConv1d / SqueezeExcite. With a fixed input every mask is a constant,
    # so: (1) polygraphy constant-fold (onnxsim 0.6.5 SIGBUS-crashes on this
    # graph on macOS-arm64 — do not use it); (2) rewrite the remaining
    # Where(constMask, constFill, x) as x*keep + fill (pure Mul/Add — NPU
    # compilers prefer it); (3) fold again; (4) strip float->float no-op Casts;
    # (5) pin the output dims the tracer left symbolic.
    import onnx
    from onnx import helper, numpy_helper
    from polygraphy.backend.onnx import fold_constants

    model = fold_constants(onnx.load(OUT_ONNX))

    inits = {i.name: i for i in model.graph.initializer}
    new_nodes, n_where = [], 0
    for n in model.graph.node:
        if n.op_type == "Where" and n.input[0] in inits and n.input[1] in inits:
            cond = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
            fill = numpy_helper.to_array(inits[n.input[1]]).astype(np.float32)
            keep_name, fill_name = n.output[0] + "_keep", n.output[0] + "_fill"
            model.graph.initializer.extend([
                numpy_helper.from_array((1.0 - cond).astype(np.float32), keep_name),
                numpy_helper.from_array((fill * cond).astype(np.float32), fill_name),
            ])
            mul = helper.make_node("Mul", [n.input[2], keep_name],
                                   [n.output[0] + "_mul"], name=n.name + "_mul")
            if np.any(fill * cond != 0):
                new_nodes += [mul, helper.make_node(
                    "Add", [mul.output[0], fill_name], [n.output[0]], name=n.name + "_add")]
            else:
                mul.output[0] = n.output[0]
                new_nodes.append(mul)
            n_where += 1
        else:
            new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model = fold_constants(model)

    # strip float->float no-op Casts (traced masked_fill leftovers)
    FLOAT = onnx.TensorProto.FLOAT
    casts = {n.output[0]: n.input[0] for n in model.graph.node
             if n.op_type == "Cast" and n.attribute[0].i == FLOAT}
    kept = []
    for n in model.graph.node:
        if n.op_type == "Cast" and n.output[0] in casts:
            continue
        for i, inp in enumerate(n.input):
            if inp in casts:
                n.input[i] = casts[inp]
        kept.append(n)
    del model.graph.node[:]
    model.graph.node.extend(kept)

    out_vi = model.graph.output[0]
    for d, v in zip(out_vi.type.tensor_type.shape.dim, (1, N_FRAMES // 8, 45)):
        d.ClearField("dim_param")
        d.dim_value = v
    print(f"cleanup: {n_where} Where nodes rewritten, {len(casts)} no-op Casts "
          f"stripped, {len(model.graph.node)} nodes total")
    ops = sorted({n.op_type for n in model.graph.node})
    print("final op set:", ops)
    banned = set(ops) & {"NonZero", "Loop", "If", "Shape", "Where", "Expand",
                         "ConstantOfShape", "Equal", "Less", "Not"}
    assert not banned, f"dynamic/mask ops survived: {banned}"
    onnx.save(model, OUT_ONNX)

    # ---- confirm NO dynamic dims survived ----
    model = onnx.load(OUT_ONNX)
    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]
        print(vi.name, dims)
        assert all(isinstance(d, int) and d > 0 for d in dims), f"dynamic dim in {vi.name}"

    # ---- onnxruntime parity ----
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT_ONNX, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"audio": wav.numpy().astype(np.float32)})[0]
    ort_err = np.abs(out - got_lp.numpy()).max()
    print(f"onnxruntime max|diff| vs torch wrapper:  {ort_err:.3e}")
    # fp32 accumulation-order noise in log-prob space; ~1e-3 observed
    assert ort_err < 5e-3

    np.save("sample_input.npy", np.random.rand(1, N_SAMPLES).astype(np.float32))
    print(f"wrote {OUT_ONNX} and sample_input.npy — input float32[1,{N_SAMPLES}], "
          f"output float32[1,64,45]")


if __name__ == "__main__":
    main()
