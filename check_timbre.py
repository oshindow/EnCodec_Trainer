utt1 = '4267_287369_000038_000000.npy'
utt2 = '6000_55211_000083_000002.npy'
utt2 = '8770_295465_000052_000000.npy'
timbre_path_1 = '/data2/junchuan/VALLE-X/timbre_vec/' + utt1
timbre_path_2 = '/data2/junchuan/VALLE-X/prosody_vec/' + utt2

timbre_path_1 = '/data2/junchuan/VALLE-X/timbre_vec/4830_25898_000008_000001.npy'
timbre_path_2 = '/data2/junchuan/VALLE-X/timbre_vec/4830_25898_000006_000004.npy'

import numpy as np

import torch
import torch.nn as nn
timbre1 = np.load(timbre_path_1)
timbre2 = np.load(timbre_path_2)
print(np.max(timbre1), np.min(timbre1))
mse_loss = nn.MSELoss()
print(timbre1.shape, timbre2.shape)
loss = mse_loss(torch.FloatTensor(timbre1), torch.FloatTensor(timbre2))
print(loss)
