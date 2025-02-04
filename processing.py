import torch
import random
import numpy as np
import torch.nn.functional as F

class CodecTokenDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tensor_cut=8, kmeans=False):
        self.data_path = data_path
        
        self.data = self.load_data()
        self.tensor_cut = tensor_cut

        # self.lengths_dict = self.get_lengths()
        self.kmeans = kmeans

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
        
        target = np.load(target_path) # [1, 491]
        target = target[:,0,:]

        timbre = np.load(timbre_path)
        content = np.load(content_path, allow_pickle=True) # [1, 326, 1024]
        print(len(content), target.shape)
       
        if self.tensor_cut:
            content_cut_length = 16000 * self.tensor_cut // 320
            target_cut_length = 24000 * self.tensor_cut // 320
            
            if target.shape[-1] > target_cut_length:
                
                start_target = random.randint(0, target.shape[-1] - target_cut_length - 1)
                start_content = int(start_target // 1.5)
                target = target[:, start_target:int(start_target+target_cut_length)]
                if self.kmeans:
                    content = content[:start_content:int(start_content+content_cut_length)]
                else:
                    content = content[:, start_content:int(start_content+content_cut_length),:]

        return uid, timbre, content, target

class CodecTokenDataCollatorWhithPadding:
    def __call__(self, batch):
        B = len(batch)
        
        # timbre: torch.Size([1, 512])
        # content: torch.Size([1, 326, 1024])
        # target: torch.Size([1, 491])

        max_tgt_length = max([item[3].shape[-1] for item in batch]) + 1
        if isinstance(batch[0][2], list):   
            max_src_length = max([len(item[2]) for item in batch])
        else:
            max_src_length = max([item[2].shape[-2] for item in batch])
        # print("max length:", max_length)
        tim_nfeats = 512 # batch[0][2].shape[-1]
        con_nfeats = 1024
        start_idx = 1024
        pad_idx = 1025
        end_idx = 1026 # 

        src_lengths = []
        tgt_lengths = []
        tims = []
        cons = []
        dec_inputs = []
        dec_outputs = []
       
        for i, item in enumerate(batch):
            tim_, con_, tar_ = item[1], item[2], item[3]
            # print(tim_.shape, con_.shape, tar_.shape)
            # (1, 512) (1, 326, 1024) (1, 491)
            tgt_lengths.append(tar_.shape[-1] + 1)
            con_len = len(con_) if isinstance(con_, list) else con_.shape[-2]
            src_lengths.append(con_len)
       
            dec_input_ = np.pad(tar_, ((0, 0), (1, 0)), constant_values=start_idx)
            dec_output_ = np.pad(tar_, ((0, 0), (0, 1)), constant_values=end_idx)
            
            dec_inputs.append(torch.tensor(np.pad(dec_input_, ((0, 0), (0, max_tgt_length - tgt_lengths[-1])), constant_values=pad_idx)))
            dec_outputs.append(torch.tensor(np.pad(dec_output_, ((0, 0), (0, max_tgt_length - tgt_lengths[-1])), constant_values=pad_idx)))
            
            if isinstance(con_, list):
                con_ = np.array(con_)[np.newaxis, :]
                cons.append(torch.tensor(np.pad(con_, ((0, 0), (0, max_src_length - con_len)), constant_values=0)).long())
            else:
                cons.append(torch.tensor(np.pad(con_, ((0, 0), (0, max_src_length - con_len), (0, 0)), constant_values=0)))
               
            tim_ = tim_[:, np.newaxis, :].repeat(con_len, axis=1)
            tims.append(torch.tensor(np.pad(tim_, ((0, 0), (0, max_src_length - tim_.shape[-2]), (0, 0)), constant_values=0)))

        # dec_input: <sos> + tokens + paddings
        # dec_output: tokens + <eos> + paddings
        src_lengths = torch.tensor(src_lengths)
        tgt_lengths = torch.tensor(tgt_lengths)

        tim = torch.concat(tims)
        con = torch.concat(cons)
        dec_input = torch.concat(dec_inputs)
        dec_output = torch.concat(dec_outputs)

        batch = {
            "tim": tim, 
            "con": con, 
            "dec_input": dec_input, 
            "dec_output": dec_output, 
            "src_lengths": src_lengths, 
            "tgt_lengths": tgt_lengths
        }
        return batch