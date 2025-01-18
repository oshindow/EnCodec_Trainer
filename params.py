class Config:
    learning_rate = 1e-3
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_steps = 5000
    batch_size = 8
    num_worker = 2
    num_train_epochs = 100
    gradient_accumulation_steps = 1
    train_path = 'data/train.txt'
    val_path = 'data/eval.txt'

    ## ARTransformer-related
    vocab_size = 1024
    input_dim = 512
    d_model = 1024
    nhead = 16
    num_encoder_layers = 12
    num_decoder_layers = 12
    dim_feedforward = 4096
    max_seq_length = 1000
    start_idx = 1024
    pad_idx = 1025
    end_idx = 1026
