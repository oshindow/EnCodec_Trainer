import params
import torch
from torch.utils.data import DataLoader
from data_utils import DistributedBucketSampler
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import customAudioDataset as data
import torch.nn.functional as F
from transformer_adapter import ARTransformer

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
params.batch_size = 8
random_seed = params.seed
params.learning_rate = 1e-4
params.max_epoch = 800
params.log_interval = 1
SAVE_LOCATION = '/data2/xintong/encodec_models/exp_E2/'
import warnings
warnings.filterwarnings("ignore")

def collate_fn(batch):
    B = len(batch)
    
    # rpsody: [1, 212, 1024]
    # timbre: [1, 512]
    # content: [1, 212, 1024]
    # target: [1, 212]

    max_length = max([item[1].shape[-2] for item in batch])
    
    pro_nfeats = batch[0][1].shape[-1]
    tim_nfeats = 512 # batch[0][2].shape[-1]
    con_nfeats = 1024

    pro = torch.zeros((B, max_length, pro_nfeats), dtype=torch.float32)
    tar = torch.ones((B, max_length), dtype=torch.float32) * -100
    tim = torch.zeros((B, max_length, tim_nfeats), dtype=torch.float32)
    con = torch.zeros((B, max_length, con_nfeats), dtype=torch.float32)
    lengths = []

    for i, item in enumerate(batch):
        pro_, tim_, con_, tar_ = item[1], item[2], item[3], item[4]
        
        lengths.append(pro_.shape[-2])

        pro[i,:pro_.shape[-2],:] = pro_
        tar[i,:tar_.shape[-1]] = tar_
        con[i,:con_.shape[-2],:] = con_
        tim[i,:tar_.shape[-2],:] = tim_

    lengths = torch.LongTensor(lengths)
    tar = F.pad(tar, (1, 0), "constant", 1024)

    return pro, tim, con, tar, lengths

def main(params):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '60001'

    params.batch_size = params.batch_size // n_gpus
    print('Batch size per GPU :', params.batch_size)

    mp.spawn(run, nprocs=n_gpus, args=(n_gpus,))

def run(rank, n_gpus):
    dist.init_process_group(
        backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    if rank == 0:
        print('Set devices ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if n_gpus > 1:
        device = torch.device("cuda:{:d}".format(rank))

    if rank == 0:
        print('Initializing data loaders...')

    train_dataset = data.CustomAudioDataset('train.txt', tensor_cut=800)
    # train_sampler = DistributedBucketSampler(
    #     # logger,
    #     train_dataset,
    #     params.batch_size,
    #     [0, 100, 400, 500, 600,700,800, 900, 1000,2500],
    #     num_replicas=n_gpus,
    #     rank=rank,
    #     shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn,)
    # trainloader = DataLoader(dataset=train_dataset,
    #                     collate_fn=collate_fn,
    #                     num_workers=8, shuffle=False, batch_sampler=train_sampler)

    import pdb
    pdb.set_trace



    model = ARTransformer()
    model.cuda(rank)
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)
    
    def train(epoch, loader):
        loader.batch_sampler.set_epoch(epoch)
        
        print('----------------------------------------Epoch: {}----------------------------------------'.format(epoch))
        for batch_idx, batch in enumerate(trainloader):

            pro, tim, con, tar, lengths = batch
            
            pro = pro.cuda(rank)
            lengths = lengths.cuda(rank)
            tim = tim.cuda(rank)
            con = con.cuda(rank)
            tar = tar.cuda(rank)
            
            # optimizer.zero_grad()
            # model.zero_grad()
            
            loss = model(pro, tim, con, tar, lengths)
            
            loss.backward()
            optimizer.step()

            if rank == 0 and (batch_idx % params.log_interval == 0):
                print(torch.cuda.mem_get_info())
                print(f"Train Epoch: {epoch} steps: {batch_idx} / {int(len(trainloader.dataset) / params.batch_size)} diff loss: {diff_loss} MSE loss: {mse_loss}")


    def adjust_learning_rate(optimizer, epoch):
        if epoch % 80 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1


    for epoch in range(1, params.max_epoch + 1):

        train(epoch, trainloader)
        torch.save(model.state_dict(), f'{SAVE_LOCATION}epoch{epoch}.pth') #epoch{epoch}.pth
        
        adjust_learning_rate(optimizer, epoch)
        

if __name__ == "__main__":
    main(params)