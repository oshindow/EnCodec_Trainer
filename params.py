class Config:
    learning_rate = 1e-5
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 10000
    batch_size = 8
    num_worker = 2
    num_train_epochs = 1000
    gradient_accumulation_steps = 1
    train_path = 'data/kmeans/train.txt'
    val_path = 'data/kmeans/eval.txt'

    ## ARTransformer-related
    vocab_size = 1024
    input_dim = 512
    d_model = 1024
    nhead = 8 # 16
    num_encoder_layers = 6 # 12
    num_decoder_layers = 6 # 12
    dim_feedforward = 2048 # 4096
    max_seq_length = 1000
    start_idx = 1024
    pad_idx = 1025
    end_idx = 1026
