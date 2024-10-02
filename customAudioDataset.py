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
        self.tensor_cut = tensor_cut

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
        
        import random

        # List of folder names
        folders = {
            0: 'prosody_vec',
            1: 'prosody_vec_200n',
            2: 'prosody_vec_400n',
            3: 'prosody_vec_200p',
            4: 'prosody_vec_400p'
        }

        # Generate a random integer between 0 and 4
        random_int = random.randint(0, 4)

        # Select the folder based on the random integer
        selected_folder = folders[random_int]
        timbre_path = timbre_path.replace('prosody_vec', selected_folder)
        prosody = torch.FloatTensor(np.load(prosody_path)).unsqueeze(0)
        timbre = torch.FloatTensor(np.load(timbre_path))
        target = torch.FloatTensor(np.load(target_path)).transpose(1, 2)[:,:prosody.shape[-2],:]

        # print(uid, prosody.shape, timbre.shape, target.shape)

        # long audio
        if self.tensor_cut > 0:
            if target.size()[-1] > self.tensor_cut:
                start = random.randint(0, target.size()[1]-self.tensor_cut-1)
                target = target[:, start:start+self.tensor_cut,:]
        return uid, prosody, timbre, target

