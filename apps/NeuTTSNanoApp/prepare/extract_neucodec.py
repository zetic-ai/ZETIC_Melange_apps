import librosa
import torch
import torchaudio
from torchaudio import transforms as T
from neucodec import NeuCodec
 
model = NeuCodec.from_pretrained("neuphonic/neucodec")
model.eval().cpu()  # Use CPU for tracing (GPU not required)   
 
# Use librosa to load audio (avoids torchcodec dependency)
audio_path = librosa.ex("libri1")
y, sr = librosa.load(audio_path, sr=None, mono=True)  # Load as mono
# Convert to torch tensor: librosa returns (samples,) for mono
y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0)  # (1, 1, T) -> (B, channels, T)

if sr != 16_000:
    y = T.Resample(sr, 16_000)(y)  # (B, 1, T_16)

with torch.no_grad():
    fsq_codes = model.encode_code(y)
    # fsq_codes = model.encode_code(librosa.ex("libri1")) # or directly pass your filepath!
    print(f"Codes shape: {fsq_codes.shape}")  
    recon = model.decode_code(fsq_codes).cpu() # (B, 1, T_24)

torchaudio.save("reconstructed.wav", recon[0, :, :], 24_000)
