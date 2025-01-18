import params
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import customAudioDataset as data
import torch.nn.functional as F
from transformer_adapter import ARTransformer
import time

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

params.batch_size = 2
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

    return pro, tim, con, tar, lengths

def main(params):
    assert torch.cuda.is_available(), "CPU training is not allowed."

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing data loaders...')

    train_dataset = data.CustomAudioDataset('train.txt', tensor_cut=800)
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

    criterion = torch.nn.CrossEntropyLoss(ignore_index=params.pad_idx)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    def train(epoch, loader):
        print(f'----------------------------------------Epoch: {epoch}----------------------------------------')
        epoch_loss = 0.
        batchn = 0.
        total_loss = 0.
        model.train()

        for batch_idx, batch in enumerate(loader):

            pro, tim, con, tar, lengths = batch

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
            pred = model(pro, tim, con, tar[:,:-1], lengths)
            loss = criterion(pred.view(-1, params.vocab_size), tar[:,1:].reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            epoch_loss += loss.item()

            log_interval = params.log_interval

            if batch_idx % log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / log_interval

                print(f"Train Epoch: {epoch} steps: {batch_idx} / {int(len(trainloader.dataset) / params.batch_size)} loss: {cur_loss}")

                total_loss = 0

        return total_loss / batchn

    for epoch in range(1, params.max_epoch + 1):
        epoch_start_time = time.time()
        epoch_loss = train(epoch, trainloader)
        torch.save(model.state_dict(), f'{SAVE_LOCATION}epoch{epoch}.pth')

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time), epoch_loss))
        print('-' * 89)

        scheduler.step()
        

if __name__ == "__main__":
    main(params)
