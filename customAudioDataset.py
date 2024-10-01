import os
import pandas as pd
import torch
import torchaudio
import random
import numpy as np


class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, tensor_cut=0, fixed_length=None):
        self.data_path = data_path
        
        self.data = []
        self.read_txt()
        
        # self.transform = transform
        # self.fixed_length = fixed_length
        # self.tensor_cut = tensor_cut

    def read_txt(self):
        # data = {}
        with open(self.data_path, 'r', encoding='utf8') as input:
            for line in input:
                uid, prosody_path, timbre_path, target_path = line.strip().split('|')
                
                self.data.append([uid, prosody_path, timbre_path, target_path])

    def __len__(self):
        # if self.fixed_length:
        #     return self.fixed_length
        return len(self.data)

    def __getitem__(self, idx):
        
        uid, prosody_path, timbre_path, target_path = self.data[idx]
        prosody = torch.FloatTensor(np.load(prosody_path)).unsqueeze(0)
        timbre = torch.FloatTensor(np.load(timbre_path))
        target = torch.FloatTensor(np.load(target_path)).transpose(1, 2)[:,:prosody.shape[-2],:]

        # print(uid, prosody.shape, timbre.shape, target.shape)

        # long audio
        # if self.tensor_cut > 0:
        #     if waveform.size()[1] > self.tensor_cut:
        #         start = random.randint(0, waveform.size()[1]-self.tensor_cut-1)
        #         waveform = waveform[:, start:start+self.tensor_cut]
        return uid, prosody, timbre, target

