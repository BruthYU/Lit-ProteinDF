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

class MethodCallback(Callback):
    def __init__(self, method_name):
        super().__init__()
        self.method_name = method_name

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.method_name in ['framediff', 'foldflow']:
            trainer.train_dataloader.sampler.add_epoch()


def load_callbacks(conf):
    callback_list = []
    # Checkpoint Callback
    callback_list.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best-{epoch:02d}-{val_loss:.3f}',
        save_top_k=2,
        mode='min',
        save_last=True,
        every_n_epochs=conf.experiment.ckpt_freq
    ))
    # Learning Rate Callback
    if conf.experiment.lr_scheduler:
        callback_list.append(plc.LearningRateMonitor(
            logging_interval=None))
    # Epoch callback
    callback_list.append(MethodCallback(conf.method_name))
    return callback_list


@hydra.main(version_base=None, config_path="config", config_name="train")
def run(conf: DictConfig) -> None:

    pl.seed_everything(conf.experiment.seed)

    data_interface = DInterface(conf)
    data_interface.datamodule.setup()
    model_interface = MInterface(conf)

    gpu_count = torch.cuda.device_count()
    conf.experiment.steps_per_epoch = math.ceil(len(data_interface.datamodule.trainset)
                                                / conf.experiment.batch_size / gpu_count)
    LOG.info(f"steps_per_epoch {conf.experiment.steps_per_epoch},  gpu_count {gpu_count}, batch_size {conf.experiment.batch_size}")

    trainer_config = {
        'devices': -1,  # Use all available GPUs
        # 'precision': 'bf16',  # Use 32-bit floating point precision
        'precision': '32',
        'max_epochs': conf.experiment.num_epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        # "strategy": 'ddp',
        "accumulate_grad_batches": 1,
        'accelerator': 'cuda',  
        'callbacks': load_callbacks(conf),
    }

    trainer = Trainer(**trainer_config)
    trainer.fit(model_interface.model, data_interface.datamodule)
    print(trainer_config)


if __name__ == '__main__':
    run()