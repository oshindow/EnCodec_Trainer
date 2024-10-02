import os
import pandas as pd
import torch
import torchaudio
import random
import numpy as np
import json

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
        prosody_path = prosody_path.replace('prosody_vec', selected_folder)
        prosody = torch.FloatTensor(np.load(prosody_path)).unsqueeze(0)
        timbre = torch.FloatTensor(np.load(timbre_path))
        target = torch.FloatTensor(np.load(target_path)).transpose(1, 2)[:,:prosody.shape[-2],:]

        # print(uid, prosody.shape, timbre.shape, target.shape)

        # long audio
        if self.tensor_cut > 0:
            # print(target.size())
            if target.size()[-2] > self.tensor_cut:
                start = random.randint(0, target.size()[1]-self.tensor_cut-1)
                target = target[:, start:start+self.tensor_cut,:]
        return uid, prosody, timbre, target

