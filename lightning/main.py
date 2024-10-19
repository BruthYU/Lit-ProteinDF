import datetime
import os
import sys
os.environ["WANDB_API_KEY"] = "97202a52488fcf2762c99ff8c68367f9bc5d4033"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import math
import argparse
import shutil
import pytorch_lightning as pl
import torch
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
from pytorch_lightning.callbacks import Callback
from model import MInterface
from data import DInterface
import logging

LOG = logging.getLogger(__name__)

class SamplerCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trainer.datamodule.train_sampler.add_epoch()



def load_callbacks(conf):
    callbacks = []

    # Checkpoint Callback
    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=5,
        mode='min',
        save_last=True,
        every_n_epochs=conf.experiment.ckpt_freq
    ))

    # Learning Rate Callback
    if conf.experiment.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))

    # Epoch as the sampler random state
    if conf.dataset.name in ['framediff', 'foldflow']:
        callbacks.append(SamplerCallback())

    return callbacks


@hydra.main(version_base=None, config_path="config", config_name="config")
def run(conf: DictConfig) -> None:


    pl.seed_everything(conf.experiment.seed)
    
    data_module = DInterface(conf)
    data_module.setup()
    gpu_count = torch.cuda.device_count()
    conf.experiment.steps_per_epoch = math.ceil(len(data_module.trainset) / conf.experiment.batch_size / gpu_count)
    LOG.info(f"steps_per_epoch {conf.experiment.steps_per_epoch},  gpu_count {gpu_count}, batch_size {conf.experiment.batch_size}")

    
    model = MInterface(conf)
    
    trainer_config = {
        'devices': -1,  # Use all available GPUs
        # 'precision': 'bf16',  # Use 32-bit floating point precision
        'precision': '32',
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'reload_dataloaders_every_n_epochs': 1,
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": 'ddp',
        "accumulate_grad_batches": 1,
        'accelerator': 'cuda',  
        'callbacks': load_callbacks(args),
        'logger': [
                    plog.WandbLogger(
                    project = 'TokenDiff',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    id = args.ex_name.replace('/', '-',5),
                    entity = "gmondy",
                    ),
                   plog.CSVLogger(args.res_dir, name=args.ex_name)],
         'gradient_clip_val': 0.5
    }

    trainer = Trainer(**trainer_config)
    
    trainer.fit(model, data_module)
    
    print(trainer_config)


if __name__ == '__main__':
    run()