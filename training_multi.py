import params
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from data_utils import DistributedBucketSampler
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import customAudioDataset as data
# from model import EncodecModel 
from utils import fix_len_compatibility
# from 
# import modules as m
from modules.encoder import SEANetEncoder

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
params.batch_size = 8
random_seed = params.seed
params.learning_rate = 1e-4
params.max_epoch = 800
params.log_interval = 1
SAVE_LOCATION = '/data2/xintong/encodec_models/exp/'

def collate_fn(batch):
    B = len(batch)
    
    # rpsody: [1, 329, 1024]
    # timbre: [1, 512]
    # target: [1, 329, 128]

    max_length = max([item[1].shape[-2] for item in batch])
    max_length = fix_len_compatibility(max_length)
    # tar_max_length = max([item[3].shape[-2] for item in batch])

    pro_nfeats = batch[0][1].shape[-1]
    tim_nfeats = 512 # batch[0][2].shape[-1]
    tar_nfeats = batch[0][3].shape[-1]

    # print(type(B), type(pro_max_length), type(pro_nfeats))
    pro = torch.zeros((B, max_length, pro_nfeats), dtype=torch.float32)
    tar = torch.zeros((B, max_length, tar_nfeats), dtype=torch.float32)
    tim = torch.zeros((B, max_length, tim_nfeats), dtype=torch.float32)

    lengths = []

    for i, item in enumerate(batch):
        pro_, tim_, tar_ = item[1], item[2], item[3]
        # if target.size()[-2] > 1000:
        #     # print('cut', target.size())
        #     start = random.randint(0, target.size()[1]-self.tensor_cut-1)
        #     target = target[:, start:start+self.tensor_cut,:]
        #     # print(target.size())
        #     self.lengths[idx] = self.tensor_cut

        lengths.append(pro_.shape[-2])
        # tar_lengths.append(tar_.shape[-2])

        pro[i,:pro_.shape[-2],:] = pro_
        tar[i,:tar_.shape[-2],:] = tar_
        tim[i,:tar_.shape[-2],:] = tim_
    
    # tim = tim.expand(-1, )
    lengths = torch.LongTensor(lengths)
    # tar_lengths = torch.LongTensor(tar_lengths)

    # print(pro.shape, tim.shape, tar.shape, lengths)
    return pro, tim, tar, lengths

def main(params):
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '60001'

    batch_size = params.batch_size // n_gpus
    print('Total batch size:', params.batch_size)
    print('Batch size per GPU :', batch_size)

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
        print('Initializing logger...')
    
    # logger = SummaryWriter(log_dir=log_dir)
    
    if rank == 0:
        print('Initializing data loaders...')

    train_dataset = data.CustomAudioDataset('train.txt', tensor_cut=1000)
    train_sampler = DistributedBucketSampler(
        # logger,
        train_dataset,
        params.batch_size,
        [0, 100, 400, 500, 600,700,800, 900, 1000,2500],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn,)
    trainloader = DataLoader(dataset=train_dataset,
                        collate_fn=collate_fn,
                        num_workers=8, shuffle=False, batch_sampler=train_sampler)
    
    # cudnn.benchmark = True

    target_bandwidths = [1.5, 3., 6, 12., 24.]
    sample_rate = 24_000
    channels = 1
    # model = EncodecModel._get_model(
    #             target_bandwidths, sample_rate, channels,
    #             causal=False, model_norm='time_group_norm', audio_normalize=True,
    #             segment=1., name='disentangle_encodec_24khz').encoder
    model = SEANetEncoder()
    # loader.batch_sampler.set_epoch(epoch)
    # model.train_quantization = False
    model.cuda(rank)
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)
    # disc = MultiScaleSTFTDiscriminator(filters=32)
    # disc.train()
    # disc.cuda()


    lr = 0.01
    # optimizer = optim.SGD([{'params': model.parameters(), 'lr': lr}], momentum=0.9)
    # optimizer_disc = optim.SGD([{'params': disc.parameters(), 'lr': lr*10}], momentum=0.9)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)
    # imizer_disc = optim.AdamW([{'params': disc.parameters(), 'lr': lr}], betas=(0.8, 0.99))

    def train(epoch, loader):
        loader.batch_sampler.set_epoch(epoch)
        last_loss = 0
        # train_d = False
        print('----------------------------------------Epoch: {}----------------------------------------'.format(epoch))
        for batch_idx, batch in enumerate(trainloader):

            pro, tim, tar, lengths = batch
            # torch.Size([5, 754, 1024]) torch.Size([5]) torch.Size([5, 512]) torch.Size([5, 754, 128]) torch.Size([5]
            # torch.Size([5, 1376, 1024]) tensor([ 259,   98, 1376,  598, 1250]) torch.Size([5, 512]) torch.Size([5, 1376, 128]) tensor([ 259,   98, 1376,  598, 1250])
            pro = pro.cuda()
            lengths = lengths.cuda()
            tim = tim.cuda()
            # tim_lengths = tim_lengths.cuda()
            tar = tar.cuda()
            # tar_lengths = tar_lengths.cuda()


            optimizer.zero_grad()
            model.zero_grad()
            # optimizer_disc.zero_grad()
            # disc.zero_grad()
            # print(tim.shape, pro.shape, lengths)
            # torch.Size([5, 643, 512]) torch.Size([5, 643, 1024]) tensor([102, 169, 643, 164, 319], device='cuda:0')
            diff_loss, xt = model(pro, tim, tar, lengths)
            diff_loss.backward()
            optimizer.step()

            if batch_idx % params.log_interval == 0:
                print(torch.cuda.mem_get_info())
                # print(f"Train Epoch: {epoch} [{batch_idx * len(input_wav)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]")
                print(f"Train Epoch: {epoch} steps: {batch_idx} / {len(trainloader.dataset) / params.batch_size} diff loss: {diff_loss}")


    def adjust_learning_rate(optimizer, epoch):
        if epoch % 80 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1


    for epoch in range(1, params.max_epoch + 1):

        train(epoch, trainloader)
        torch.save(model.state_dict(), f'{SAVE_LOCATION}epoch{epoch}.pth') #epoch{epoch}.pth
        # torch.save(disc.state_dict(), f'{SAVE_LOCATION}epoch{epoch}_disc.pth')

        adjust_learning_rate(optimizer, epoch)
        # adjust_learning_rate(optimizer_disc, epoch)


if __name__ == "__main__":
    main(params)