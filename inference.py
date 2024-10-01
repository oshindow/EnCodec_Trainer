# from utils.generation import SAMPLE_RATE, generate_audio, preload_models, audio_rec
from scipy.io.wavfile import write as write_wav
# from IPython.display import Audio
import time
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)

import torch
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
import os
import numpy as np

device = torch.device("cuda", 0)
codec = AudioTokenizer(device)

wavroot = '/data2/xintong/tts/LibriTTS/train-clean-100'
for root, dirs, files in os.walk(wavroot):
    for file in files:
        if '.wav' not in file:
            continue
        wavefile = os.path.join(root, file)
        # wavefile = '/home/xintong/VALL-E-X/evaluation/TTS/libritts/wavs/174_50561_000013_000000.wav'
        wav_pr, sr = torchaudio.load(wavefile)
        # print(sr, wav_pr.shape)
        wav = wav_pr
        if not isinstance(wav, torch.FloatTensor):
            wav = torch.tensor(wav)
        if wav.abs().max() > 1:
            wav /= wav.abs().max()
        if wav.size(-1) == 2:
            wav = wav.mean(-1, keepdim=False)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        assert wav.ndim and wav.size(0) == 1
        wav = convert_audio(wav, sr, codec.sample_rate, codec.channels)
        wav = wav.unsqueeze(0).to(device)

        # Extract discrete codes from EnCodec
        with torch.no_grad():
            encoded_frames = codec.codec.forward(wav)

        savepath = os.path.join(root, file[:-4] + '.npy').replace('LibriTTS', 'LibriTTS_encodec_continuous')
        if not os.path.exists(os.path.dirname(savepath)):
            os.makedirs(os.path.dirname(savepath))
        np.save(savepath, encoded_frames.cpu().numpy())

