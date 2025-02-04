import os
import torch

from pathlib import Path
from transformer_adapter import ARTransformer
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from params import Config

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


if __name__ == '__main__':
    
    train_name = "lightning"
    train_id = "00007"

    log_output_dir = "exp" 
    check_output_dir = "exp" 

    cfg = Config()

    Path(log_output_dir).mkdir(parents=True, exist_ok=True)
    Path(check_output_dir).mkdir(parents=True, exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir,
        name=train_name,
        version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1 # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    model = ARTransformer(
        cfg=cfg,
        vocab_size = cfg.vocab_size,
        input_dim = cfg.input_dim,
        d_model = cfg.d_model,
        nhead = cfg.nhead,
        num_encoder_layers = cfg.num_encoder_layers,
        num_decoder_layers = cfg.num_decoder_layers,
        dim_feedforward = cfg.dim_feedforward,
        max_seq_length = cfg.max_seq_length,
        pad_idx=cfg.pad_idx
    )
    
    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
        val_check_interval=0.3
    )

    trainer.fit(model)