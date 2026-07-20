#!/usr/bin/env python3
"""Reference log-mel (pure stdlib) replicating openai-whisper's log_mel_spectrogram:
 torch.stft(n_fft=400, hop=160, periodic Hann, center=True/reflect),
 power=|stft[:-1]|^2, mel=filters@power, log10, clamp max-8, (x+4)/4.
Prints golden sample points for the Dart log_mel_test. Uses the SAME filterbank
asset the app bundles, so the test validates the Dart STFT+matmul+scale path."""
import math, struct, sys

SR=16000; N_FFT=400; HOP=160; N_MELS=80; N_FREQS=N_FFT//2+1

def load_filters(path):
    with open(path,'rb') as f: data=f.read()
    vals=struct.unpack('<%df'%(N_MELS*N_FREQS), data)
    return [list(vals[i*N_FREQS:(i+1)*N_FREQS]) for i in range(N_MELS)]

def reflect_pad(a,p):
    left=[a[i] for i in range(p,0,-1)]
    right=[a[len(a)-2-i] for i in range(p)]
    return left+list(a)+right

def logmel(audio, W):
    pad=N_FFT//2
    padded=reflect_pad(audio,pad)
    win=[0.5-0.5*math.cos(2*math.pi*n/N_FFT) for n in range(N_FFT)]
    nfr=1+(len(padded)-N_FFT)//HOP
    nfr-=1  # drop last (whisper stft[..., :-1])
    # precompute twiddles
    cos=[[math.cos(-2*math.pi*k*n/N_FFT) for n in range(N_FFT)] for k in range(N_FREQS)]
    sin=[[math.sin(-2*math.pi*k*n/N_FFT) for n in range(N_FFT)] for k in range(N_FREQS)]
    melspec=[[0.0]*nfr for _ in range(N_MELS)]
    logvals=[[0.0]*nfr for _ in range(N_MELS)]
    gmax=-1e30
    for t in range(nfr):
        s=t*HOP
        seg=[padded[s+n]*win[n] for n in range(N_FFT)]
        power=[0.0]*N_FREQS
        for k in range(N_FREQS):
            re=0.0; im=0.0; ck=cos[k]; sk=sin[k]
            for n in range(N_FFT):
                re+=seg[n]*ck[n]; im+=seg[n]*sk[n]
            power[k]=re*re+im*im
        for m in range(N_MELS):
            wm=W[m]; acc=0.0
            for k in range(N_FREQS): acc+=wm[k]*power[k]
            v=acc if acc>1e-10 else 1e-10
            lv=math.log10(v)
            logvals[m][t]=lv
            if lv>gmax: gmax=lv
    out=[[0.0]*nfr for _ in range(N_MELS)]
    for m in range(N_MELS):
        for t in range(nfr):
            lv=logvals[m][t]
            if lv< gmax-8.0: lv=gmax-8.0
            out[m][t]=(lv+4.0)/4.0
    return out,nfr,gmax

if __name__=='__main__':
    W=load_filters(sys.argv[1])
    L=16000
    audio=[0.5*math.sin(2*math.pi*440.0*i/SR) for i in range(L)]
    out,nfr,gmax=logmel(audio,W)
    print('nframes=%d gmax_log10=%.6f'%(nfr,gmax))
    pts=[(0,0),(0,50),(1,0),(10,10),(20,30),(40,50),(79,99),(5,99)]
    for (m,t) in pts:
        print('GOLDEN m=%d t=%d val=%.6f'%(m,t,out[m][t]))
