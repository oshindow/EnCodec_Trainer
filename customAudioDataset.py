import os
import pandas as pd
import torch
import torchaudio
import random
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, tensor_cut=0, fixed_length=None):
        self.data_path = data_path
        
        self.data = []
        self.read_txt()
        # self.length = []
        # self.transform = transform
        # self.fixed_length = fixed_length
        self.tensor_cut = tensor_cut
        # if train:
        self.lengths_dict = self.get_lengths()
        # oooooooooooooooooooooo
        # self.write_lengths()
        self.lengths = [self.lengths_dict[key[0]] for key in self.data]
            # self.accents = [int(key[3]) for key in self.filelist ]
        # self.scaler = StandardScaler()
    def write_lengths(self):
        self.lengths = {}
        idx = 0
        self.lengths_max = 0
        for uid, pro, tim, tar in self.data:
            if idx and idx % 1000 == 0:
                print(idx)
            # mel_path = file[0]
            # mel = self.get_mel(mel_path)
            var = np.load(tar)
            length = var.shape[2]
            self.lengths_max = max(length, self.lengths_max) # 2494
            self.lengths[os.path.basename(tar[:-4])] = length
            idx += 1

        print(self.lengths_max)
        with open('lengths.json', 'w', encoding='utf8') as output:
            json.dump(self.lengths, output, indent=4)
            
        return self.lengths
    
    def get_lengths(self):
        with open('lengths.json', 'r', encoding='utf8') as input:
            self.lengths_dict = json.load(input)
        self.lengths_max = 2494
        return self.lengths_dict

    def read_txt(self):
        with open(self.data_path, 'r', encoding='utf8') as input:
            for line in input:
                uid, prosody_path, timbre_path, content_path, target_path = line.strip().split('|')
                
                self.data.append([uid, prosody_path, timbre_path, content_path, target_path])

    def __len__(self):
        return len(self.data)

    def norm(self, x, dim):
        
        # Find the minimum and maximum along the specified axis
        x_min = x.min(dim=dim, keepdim=True)[0]
        x_max = x.max(dim=dim, keepdim=True)[0]

        # Perform min-max normalization to the range [0, 1]
        x_normalized = (x - x_min) / (x_max - x_min)

        # Scale to the range [-1, 1]
        return 2 * x_normalized - 1, x_normalized

    def __getitem__(self, idx):
        
        uid, prosody_path, timbre_path, content_path, target_path = self.data[idx]
        
        import random

        folders = {
            0: 'prosody_vec',
            1: 'prosody_vec_200n',
            2: 'prosody_vec_400n',
            3: 'prosody_vec_200p',
            4: 'prosody_vec_400p'
        }

        random_int = random.randint(0, 4)

        selected_folder = folders[random_int]
        prosody_path = prosody_path.replace('prosody_vec', selected_folder)
        prosody_scaled = self.norm(torch.FloatTensor(np.load(prosody_path)), dim=0)[0]
        
        prosody = prosody_scaled.unsqueeze(0)
        timbre = torch.FloatTensor(np.load(timbre_path))
        target = torch.FloatTensor(np.load(target_path)).transpose(1, 2)[:,:prosody.shape[-2],0]
       
        content = torch.FloatTensor(np.load(content_path))
        new_length = target.shape[-1]
        content = F.interpolate(content.transpose(1,2), size=new_length, mode='linear', align_corners=True).transpose(1,2)
        
        if self.tensor_cut:
            if target.size()[-1] > self.tensor_cut:
                start = random.randint(0, target.size()[1]-self.tensor_cut-1)
                target = target[:, start:start+self.tensor_cut]
                prosody = prosody[:, start:start+self.tensor_cut,:]
                content = content[:, start:start+self.tensor_cut,:]
 
        return uid, prosody, timbre, content, target

