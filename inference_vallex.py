# from utils.generation import SAMPLE_RATE, generate_audio, preload_models, audio_rec
from scipy.io.wavfile import write as write_wav
# from IPython.display import Audio
import time
# from data.tokenizer import (
#     AudioTokenizer,
#     tokenize_audio,
# )

import torch
import torch.nn as nn
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
import os
import numpy as np
from modules.encoder import SEANetEncoder
from collections import OrderedDict
device = torch.device("cuda", 0)
# codec = AudioTokenizer(device)

# wavroot = '/data2/xintong/tts/LibriTTS/train-clean-100'

# 1. prosody, timbre -> encoder continues (ours) 
# 2. encoder continues -> wav (quantizer, encodec decoder)
# 3. 

# 1. 

utt = '4267_287369_000038_000000.npy'
prosody_path = '/data2/junchuan/VALLE-X/prosody_vec/' + utt
timbre_path = '/data2/junchuan/VALLE-X/timbre_vec/' + utt
target_path = '/data2/xintong/LibriTTS_encodec_continuous/train-clean-100/4267/287369/4267_287369_000038_000000.npy'

model = SEANetEncoder()
pretrained_model = '/data2/xintong/encodec_models/exp/epoch62.pth'

checkpoint = torch.load(pretrained_model)
# print(checkpoint)
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k.replace("module.", "")  # Remove 'module.' prefix
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
        
model.cuda().eval()

pro = torch.FloatTensor(np.load(prosody_path)).unsqueeze(0)
timbre = torch.FloatTensor(np.load(timbre_path)).unsqueeze(1).expand(-1,pro.shape[-2],-1)
target = torch.FloatTensor(np.load(target_path))
lengths = torch.LongTensor([timbre.shape[-2]])

pro = pro.cuda()
timbre = timbre.cuda()
lengths = lengths.cuda()
target = target.cuda()

with torch.no_grad():
    print(pro.shape, timbre.shape, lengths)
    output = model.inference(pro, timbre, lengths)

print(output.shape, output)
print(target.shape, target)
mse_loss = nn.MSELoss()
loss = mse_loss(output, target[:,:,:-1])
print(loss)
savepath = 'predict_continuous.npy'
# if not os.path.exists(os.path.dirname(savepath)):
#     os.makedirs(os.path.dirname(savepath))
np.save(savepath, output.cpu().numpy())



