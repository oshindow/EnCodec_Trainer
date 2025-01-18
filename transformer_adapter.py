

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
import math

device = torch.device("cuda:0")
    
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

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Implements the positional encoding for the Transformer model.
        Args:
            d_model (int): The dimension of the model.
            max_len (int): The maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model) with sinusoidal values
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension and register as a buffer
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Add positional encoding to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Positional encoded tensor of the same shape as x.
        """

        return x + self.pe[:, :x.size(1), :].to(device)

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
    def __init__(self, vocab_size, input_dim, max_seq_length, pad_idx, d_model=1024, nhead=16, num_encoder_layers=12, num_decoder_layers=12, dim_feedforward=4096):
        super(ARTransformer, self).__init__()

        # self.preconv_prosody = torch.nn.Sequential(
        #     nn.Conv1d(1024, input_dim, kernel_size=3, padding=1),
        #     Transpose(1,2),
        #     nn.LayerNorm(input_dim),
        #     Mish())
        self.preconv_timbre = torch.nn.Sequential(
            nn.Conv1d(512, input_dim, kernel_size=3, padding=1),
            Transpose(1,2),
            nn.LayerNorm(input_dim),
            Mish(),
            nn.Linear(512, 1024))
        self.preconv_content = torch.nn.Sequential(
            nn.Conv1d(1024, input_dim, kernel_size=3, padding=1),
            Transpose(1,2),
            nn.LayerNorm(input_dim),
            Mish())
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.embedding = nn.Embedding(vocab_size + 3, d_model, padding_idx=pad_idx)
        self.pe = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_decoder_layers
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, pro, tim, con, tgt, seq_len_list):
        # pro_emb = self.preconv_prosody(pro.permute(0, 2, 1).contiguous())
        tim_emb = self.preconv_timbre(tim.permute(0, 2, 1).contiguous())
        con_emb = self.preconv_content(con.permute(0, 2, 1).contiguous())
        src = con_emb

        src_emb = self.input_projection(src)
        src_emb = self.pe(src_emb)

        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pe(tgt_emb)

        src_key_padding_mask = generate_padding_mask(seq_len_list, src_emb.shape[1]).to(src.device)
        tgt_key_padding_mask = generate_padding_mask(seq_len_list, src_emb.shape[1]).to(tgt.device)
        tgt_mask = generate_tgt_mask(tgt_emb.shape[1]).to(tgt.device)

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask.permute(1, 0))

        memory = memory + tim_emb

        output = self.decoder(tgt_emb.permute(1, 0, 2), memory.permute(1, 0, 2), tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

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