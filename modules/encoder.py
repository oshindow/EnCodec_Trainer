from .Unet import Diffusion
import torch
import torch.nn as nn

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

class SEANetEncoder(nn.Module):    
    def __init__(self):
        super().__init__()

        self.preconv_prosody = nn.Conv1d(1024, 128, kernel_size=3, padding=1)
        self.preconv_timbre = nn.Conv1d(512, 128, kernel_size=3, padding=1)

        self.diffusion = Diffusion(n_feats=128, dim=64)
    
    def forward(self, pro, tim, target, lengths):
        pro_emb = self.preconv_prosody(pro.permute(0, 2, 1))
        tim_emb = self.preconv_timbre(tim.permute(0, 2, 1))
        emb = (pro_emb + tim_emb)
        target_mask = sequence_mask(lengths, max_length=target.shape[-2]).unsqueeze(1).to(emb)
        # print(target_mask.shape)
        #target = target.transpose(1,2)
        # target = target.permute(0,2,1)
        # print(target.shape)
        # output = self.diffusion.estimator(target, target_mask, emb)
        diff_loss, xt = self.diffusion.compute_loss(target, target_mask, emb)
        
        return diff_loss, xt