# from model.utils import fix_len_compatibility

# data parameters
train_filelist_path = 'resources/filelists/ljspeech/train.txt'
valid_filelist_path = 'resources/filelists/ljspeech/valid.txt'
test_filelist_path = 'resources/filelists/ljspeech/test.txt'
cmudict_path = 'resources/cmu_dictionary'
zhdict_path = 'resources/zh_dictionary.json'
add_blank = True
n_feats = 80
n_spks = 1  # 247 for Libri-TTS filelist and 1 for LJSpeech
spk_emb_dim = 64
n_feats = 80
n_fft = 1024
sample_rate = 16000
hop_length = 256
win_length = 1024
f_min = 80
f_max = 7600

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 12
enc_kernel = 3
enc_dropout = 0.1
n_heads = 16
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
log_dir = '/data2/xintong/gradtts/logs/new_exp'
test_size = 4
n_epochs = 500
batch_size = 4
learning_rate = 1e-2
seed = 37
save_every = 1
# out_size = fix_len_compatibility(100 * 2*16000//256)

vocab_size=1024
input_dim=512
d_model=1024
nhead=16
num_encoder_layers=12
num_decoder_layers=12
dim_feedforward=4096
max_seq_length=1000
start_idx=1024
pad_idx=1025
end_idx=1026
