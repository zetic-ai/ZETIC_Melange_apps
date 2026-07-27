#!/usr/bin/env python3
"""Generate the OpenAI/librosa 80-mel Slaney filterbank as float32 LE [80,201].
Pure stdlib (no numpy) so it runs anywhere. Reproduces
librosa.filters.mel(sr=16000, n_fft=400, n_mels=80, htk=False, norm='slaney'),
which is exactly what openai-whisper bundles in mel_filters.npz (n_mels=80).
Output: assets/mel_filters_80.bin  (80 rows x 201 cols, row-major, float32 LE).
"""
import math, struct, sys

SR=16000; N_FFT=400; N_MELS=80; FMIN=0.0; FMAX=SR/2.0
N_FREQS=N_FFT//2+1  # 201

def hz_to_mel(f):
    f_sp=200.0/3.0
    min_log_hz=1000.0; min_log_mel=(min_log_hz-0.0)/f_sp
    logstep=math.log(6.4)/27.0
    if f>=min_log_hz:
        return min_log_mel+math.log(f/min_log_hz)/logstep
    return f/f_sp

def mel_to_hz(m):
    f_sp=200.0/3.0
    min_log_hz=1000.0; min_log_mel=(min_log_hz-0.0)/f_sp
    logstep=math.log(6.4)/27.0
    if m>=min_log_mel:
        return min_log_hz*math.exp(logstep*(m-min_log_mel))
    return f_sp*m

def build():
    fftfreqs=[i*(SR/2.0)/(N_FREQS-1) for i in range(N_FREQS)]
    m_min=hz_to_mel(FMIN); m_max=hz_to_mel(FMAX)
    mels=[m_min+(m_max-m_min)*i/(N_MELS+1) for i in range(N_MELS+2)]
    freqs=[mel_to_hz(m) for m in mels]
    fdiff=[freqs[i+1]-freqs[i] for i in range(len(freqs)-1)]
    W=[[0.0]*N_FREQS for _ in range(N_MELS)]
    for i in range(N_MELS):
        enorm=2.0/(freqs[i+2]-freqs[i])
        for j in range(N_FREQS):
            lower=-(freqs[i]-fftfreqs[j])/fdiff[i]
            upper=(freqs[i+2]-fftfreqs[j])/fdiff[i+1]
            v=max(0.0,min(lower,upper))
            W[i][j]=v*enorm
    return W

if __name__=='__main__':
    W=build()
    out=sys.argv[1]
    with open(out,'wb') as f:
        for row in W:
            for v in row:
                f.write(struct.pack('<f', v))
    print('wrote',out,N_MELS,'x',N_FREQS,'=',N_MELS*N_FREQS,'floats')
