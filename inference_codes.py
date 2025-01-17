import params
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import customAudioDataset as data
import torch.nn.functional as F
from transformer_adapter import ARTransformer
import time
import math
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

params.batch_size = 1
random_seed = params.seed
params.learning_rate = 1e-4
params.max_epoch = 800
params.log_interval = 1
SAVE_LOCATION = '/data2/xintong/encodec_models/exp_E2/'

import warnings
warnings.filterwarnings("ignore")

def collate_fn(batch):
    B = len(batch)
    
    max_length = max([item[1].shape[-2] for item in batch])
    
    pro_nfeats = batch[0][1].shape[-1]
    tim_nfeats = 512
    con_nfeats = 1024

    start_idx = params.start_idx
    pad_idx = params.pad_idx

    pro = torch.zeros((B, max_length, pro_nfeats), dtype=torch.float32)
    tar = torch.ones((B, max_length), dtype=torch.float32) * pad_idx
    tim = torch.zeros((B, max_length, tim_nfeats), dtype=torch.float32)
    con = torch.zeros((B, max_length, con_nfeats), dtype=torch.float32)
    lengths = []
    
    for i, item in enumerate(batch):
        pro_, tim_, con_, tar_ = item[1], item[2], item[3], item[4]
        lengths.append(pro_.shape[-2])

        pro[i, :pro_.shape[-2], :] = pro_
        tar[i, :tar_.shape[-1]] = tar_
        con[i, :con_.shape[-2], :] = con_
        tim[i, :tar_.shape[-2], :] = tim_

    lengths = torch.LongTensor(lengths)
    tar = F.pad(tar, (1, 0), "constant", start_idx)

    return item[0], pro, tim, con, tar, lengths

def main(params):
    assert torch.cuda.is_available(), "CPU training is not allowed."

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing data loaders...')

    train_dataset = data.CustomAudioDataset('/home/xintong/EnCodec_Trainer/eval_one.txt', tensor_cut=None, training=False)
    trainloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)

    model = ARTransformer(
        vocab_size = params.vocab_size,
        input_dim = params.input_dim,
        d_model = params.d_model,
        nhead = params.nhead,
        num_encoder_layers = params.num_encoder_layers,
        num_decoder_layers = params.num_decoder_layers,
        dim_feedforward = params.dim_feedforward,
        max_seq_length = params.max_seq_length,
        pad_idx=params.pad_idx
    )
    model.to(device)
    ckpt = torch.load('/data2/xintong/encodec_models/exp_E2/epoch100.pth', map_location=device)
    
    # Get the model's state_dict from the checkpoint
    state_dict = ckpt

    # Create a new state_dict without the 'module.' prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'module.' prefix from keys
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=params.pad_idx)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def infer(loader):
        # print(f'----------------------------------------Epoch: {epoch}----------------------------------------')
        epoch_loss = 0.
        batchn = 0.
        total_loss = 0.
        model.eval()

        for batch_idx, batch in enumerate(loader):

            uid, pro, tim, con, tar, lengths = batch
            print(uid)
            pro, tim, con, tar, lengths = pro.to(device), tim.to(device), con.to(device), tar.to(device), lengths.to(device)

            if (torch.isnan(pro).any() or torch.isinf(pro).any() or
                torch.isnan(tim).any() or torch.isinf(tim).any() or
                torch.isnan(con).any() or torch.isinf(con).any() or
                torch.isnan(tar).any() or torch.isinf(tar).any()):
                
                continue

            if (tar < 0).any() or (tar > 1025).any():
                print(batch_idx)
                raise ValueError(f"tgt tensor contains values out of bounds: {tar[tar < 0]} and {tar[tar > 1025]}")

            batchn = batchn + 1
            tar = tar.to(torch.long)
            lengths = lengths.tolist()

            optimizer.zero_grad()
            with torch.no_grad():
                # pred = model(pro, tim, con, tar[:,:-1], lengths)
                # tar_one = onetime(pro, tim, con, tar, lengths)
                # tar_greedy = greedysearch(pro, tim, con, device)
                # tar_beam = beamsearch(pro, tim, con, device)
                tar_dbs = diverse_beamsearch(pro, tim, con, device)
    
            # probabilities = torch.softmax(pred, dim=-1)
            # predicted_class = torch.argmax(probabilities, dim=-1)
            # print(predicted_class.max(), predicted_class.min())
            # np.save('/home/xintong/EnCodec_Trainer/output/codes/' + uid + '_one.npy', tar_one.cpu().numpy())
            # np.save('/home/xintong/EnCodec_Trainer/output/codes/' + uid + '_greedy.npy', tar_greedy.cpu().numpy())
            # np.save('/home/xintong/EnCodec_Trainer/output/codes/' + uid + '_beam.npy', tar_beam.cpu().numpy())
            np.save('/home/xintong/EnCodec_Trainer/output/codes/' + uid + '_dbs.npy', tar_dbs.cpu().numpy())
            
        # return total_loss / batchn
    def onetime(pro, tim, con, tar, lengths):
        pred = model(pro, tim, con, tar[:,:-1], lengths)
        probabilities = torch.softmax(pred, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)
        return predicted_class
    
    def greedysearch(pro, tim, con, device):
        max_length = pro.shape[1]
        tar = torch.LongTensor([1024]).unsqueeze(0).to(device)
        src_length = [max_length]
        for i in range(max_length):
            tgt_length = [tar.shape[-1]]
            
            logits = model.inference(pro, tim, con, tar, src_length, tgt_length)  # Shape: (batch_size, seq_length, vocab_size)
            
            next_token_logits = logits[:, -1, :]  # Shape: (batch_size, length, vocab_size)
            
            next_token_probs = F.softmax(next_token_logits, dim=-1)  # Shape: (batch_size, vocab_size)
        
            next_token = torch.argmax(next_token_probs, dim=-1)  # Shape: (batch_size)
            next_token = next_token.unsqueeze(1)  # Shape: (batch_size, 1)
            tar = torch.cat([tar, next_token], dim=1)
        return tar
    
    def beamsearch(pro, tim, con, device):
        beam_width = 5  # Number of sequences to maintain in the beam
        max_length = pro.shape[1]
        tar_start = torch.LongTensor([1024]).unsqueeze(0).to(device)
        src_length = [max_length]
        
        # Initialize the beam with the starting token
        beam = [(tar_start, 0)]  # Each entry is (sequence tensor, cumulative log probability)
        
        for i in range(max_length):
            candidates = []
            
            for seq, score in beam:
                tgt_length = [seq.shape[-1]]
                logits = model.inference(pro, tim, con, seq, src_length, tgt_length)
                next_token_logits = logits[:, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)

                # sorted_next_token_probs = torch.sort(next_token_probs, descending=True).values.tolist()
                # Get top-k tokens and their probabilities
                top_k_probs, top_k_indices = next_token_probs.topk(beam_width, dim=-1)
                
                for j in range(beam_width):
                    next_token = top_k_indices[:, j].unsqueeze(1)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score * top_k_probs[0, j].item()
                    candidates.append((new_seq, new_score))
            
            # Select top-k candidates with the highest cumulative probabilities
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Final sequence with the highest score
        best_sequence = beam[0][0]
        return best_sequence
    
    def diverse_beamsearch(pro, tim, con, device):
        beam_width = 5  # Number of sequences to maintain in the beam
        max_length = pro.shape[1]
        base_penalty = 0.05  # Adjust the base penalty strength for each consecutive repetition
        
        tar_start = torch.LongTensor([1024]).unsqueeze(0).to(device)
        src_length = [max_length]
        
        # Initialize the beam with the starting token
        beam = [(tar_start, 0)]  # Each entry is (sequence tensor, cumulative log probability)
        
        count_consecutive = [0,0,0,0,0]
        for i in range(max_length):
            candidates = []
            
            # count_consecutive = 0
            # beam:  
            # [ [1024,789,], 
            #       predict 789, 780, consecutive = 1,
            #   [1024,789, 789], predict 789, consecutive = 2
            # 
            #   [1024,780,]
            # ]
            for idx, (seq, score) in enumerate(beam): 
                # beam: [(seq0, score0), (1,2,3,4)]
                tgt_length = [seq.shape[-1]]
                logits = model.inference(pro, tim, con, seq, src_length, tgt_length)
                next_token_logits = logits[:, -1, :]
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Get top-k tokens and their probabilities
                top_k_probs, top_k_indices = next_token_probs.topk(beam_width, dim=-1)
                
                prev_token = seq[0][-1]
                if prev_token.item() in top_k_indices[0].tolist():
                    count_consecutive[idx] += 1
                else:
                    count_consecutive[idx] = 0

                if count_consecutive[idx] > 0:
                    penalty = math.log(base_penalty) ** count_consecutive[idx]  # Higher count means stronger penalty
                    next_token_probs[0][prev_token.item()] += penalty
                
                # second pass
                top_k_probs, top_k_indices = next_token_probs.topk(beam_width, dim=-1)

                ## 
                for j in range(beam_width):
                    next_token = top_k_indices[:, j].unsqueeze(1)
                    new_seq = torch.cat([seq, next_token], dim=1)
                    new_score = score + top_k_probs[0, j].item()
                    candidates.append((new_seq, new_score))

            # Select top-k candidates with the highest cumulative probabilities
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Final sequence with the highest score
        best_sequence = beam[0][0]
        return best_sequence

    infer(trainloader)      
        

if __name__ == "__main__":
    main(params)
