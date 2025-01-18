import torch
import random
import numpy as np
import torch.nn.functional as F

class CodecTokenDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tensor_cut=8, training=True):
        self.data_path = data_path
        
        self.data = self.load_data()
        self.tensor_cut = tensor_cut

        self.lengths_dict = self.get_lengths()
        self.training = training


    def load_data(self):
        data = []
        with open(self.data_path, 'r', encoding='utf8') as input:
            for line in input:
                uid, prosody_path, timbre_path, content_path, target_path = line.strip().split('|')
                
                data.append([uid, prosody_path, timbre_path, content_path, target_path])
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        uid, prosody_path, timbre_path, content_path, target_path = self.data[idx]
        
        timbre = torch.FloatTensor(np.load(timbre_path))
        target = torch.FloatTensor(np.load(target_path)).transpose(1, 2)[:,:timbre.shape[-2],0]
       
        content = torch.FloatTensor(np.load(content_path))
        # new_length = target.shape[-1]
        # content = F.interpolate(content.transpose(1,2), size=new_length, mode='linear', align_corners=True).transpose(1,2)
        
        if self.tensor_cut:
            if target.size()[-1] > self.tensor_cut:
                start = random.randint(0, target.size()[1]-self.tensor_cut-1)
                target = target[:, start:start+self.tensor_cut]
                content = content[:, start:start+self.tensor_cut,:]
 
        return uid, timbre, content, target

class CodecTokenDataCollatorWhithPadding:
    def __call__(self, batch):
        B = len(batch)
        
        # rpsody: [1, 212, 1024]
        # timbre: [1, 512]
        # content: [1, 212, 1024]
        # target: [1, 212]

        max_length = max([item[1].shape[-2] for item in batch])
        
        tim_nfeats = 512 # batch[0][2].shape[-1]
        con_nfeats = 1024
        start_idx = 1024
        pad_idx = 1025
        end_idx = 1026 # 

        tar = torch.ones((B, max_length), dtype=torch.float32) * pad_idx
        tim = torch.zeros((B, max_length, tim_nfeats), dtype=torch.float32)
        con = torch.zeros((B, max_length, con_nfeats), dtype=torch.float32)
        lengths = []

        for i, item in enumerate(batch):
            pro_, tim_, con_, tar_ = item[1], item[2], item[3], item[4]
            
            lengths.append(pro_.shape[-2])

            tar[i,:tar_.shape[-1]] = tar_
            con[i,:con_.shape[-2],:] = con_
            tim[i,:tar_.shape[-2],:] = tim_

        lengths = torch.LongTensor(lengths)
        dec_output = F.pad(tar, (1, 0), "constant", start_idx)
        dec_input = F.pad(tar, (0, 1), "constant", end_idx)[1:]

        return tim, con, dec_input, dec_output, lengths