

import os
from typing import Dict
import numpy as np

import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import copy

import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import customAudioDataset as data
from encodec import EncodecModel
    
def generate_src_mask():
    return None

def generate_tgt_mask(tgt_seq_len):
    return nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(torch.bool)

def generate_padding_mask(seq_lengths, max_len):
    batch_size = len(seq_lengths)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    
    for i, seq_len in enumerate(seq_lengths):
        mask[i, :seq_len] = False 
    return mask  # Shape: [batch_size, max_len]

import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
    
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ARTransformer(nn.Module):
    def __init__(self, vocab_size, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pad_idx):
        super(ARTransformer, self).__init__()

        self.preconv_prosody = torch.nn.Sequential(
            nn.Conv1d(1024, input_dim, kernel_size=3, padding=1),
            Transpose(1,2),
            nn.LayerNorm(input_dim),
            Mish())
        self.preconv_timbre = torch.nn.Sequential(
            nn.Conv1d(512, input_dim, kernel_size=3, padding=1),
            Transpose(1,2),
            nn.LayerNorm(input_dim),
            Mish())
        self.preconv_content = torch.nn.Sequential(
            nn.Conv1d(1024, input_dim, kernel_size=3, padding=1),
            Transpose(1,2),
            nn.LayerNorm(input_dim),
            Mish())
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.embedding = nn.Embedding(vocab_size + 2, d_model, padding_idx=pad_idx)
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, pro, tim, con, tgt, seq_len_list):
        pro_emb = self.preconv_prosody(pro.permute(0, 2, 1).contiguous())
        tim_emb = self.preconv_timbre(tim.permute(0, 2, 1).contiguous())
        con_emb = self.preconv_content(con.permute(0, 2, 1).contiguous())
        src = pro_emb + tim_emb + con_emb

        src_emb = self.input_projection(src) + self.pos_encoder(torch.arange(src.size(1)).to(src.device))
        # print(tgt.max(), tgt.min())
        tgt_emb = self.embedding(tgt) + self.pos_encoder(torch.arange(tgt.size(1)).to(tgt.device))

        src_key_padding_mask = generate_padding_mask(seq_len_list, src_emb.shape[1]).to(src.device)
        tgt_key_padding_mask = generate_padding_mask(seq_len_list, src_emb.shape[1]).to(tgt.device)
        tgt_mask = generate_tgt_mask(tgt_emb.shape[1]).to(tgt.device)
        
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        return self.fc_out(output)

# net_model = ARTransformer(vocab_size=1024, 
#                 input_dim=512,
#                 d_model=256, 
#                 nhead=2, 
#                 num_encoder_layers=10, 
#                 num_decoder_layers=10, 
#                 dim_feedforward=1024, 
#                 max_seq_length=1000)

# bs = 4
# prosody = torch.randn(bs, 212, 1024)
# timbre = torch.randn(bs, 212, 512)
# content = torch.randn(bs, 212, 1024)
# target = torch.randint(0, 1024, (bs, 212))
# import random
# seq_len_list = [random.randint(200, 211) for _ in range(bs)]
# output = net_model(prosody, target, seq_len_list)