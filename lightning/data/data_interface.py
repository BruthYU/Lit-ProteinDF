import inspect
import importlib
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from lightning.utils.utils import cuda
from lightning.data.framediff.dataset import FrameDiff_Dataset

class DInterface(pl.LightningDataModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters()
        self.data_conf = conf.dataset
        self.frame_conf = conf.frame
        self.method = self.data_conf.name
        self.batch_size = self.data_conf.batch_size
        self.data_module = self.init_data_module(self.method)

        # import utils for to create dataloader
        self.dataloader = importlib.import_module(f'lightning.data.{self.method.lower()}.dataloader')
        self.device = 'cuda:0'

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize_module(module = self.data_module,is_training=True,
                                                      frame_conf=self.frame_conf, data_conf=self.data_conf)

            '''Test the trainset in dataloader'''
            train_loader = DataLoader(
                    self.trainset,
                    batch_size= 2,
                    pin_memory=True,
                    shuffle=True)


            self.valset = self.instancialize_module(module=self.data_module, is_training=False,
                                                      frame_conf=self.frame_conf, data_conf=self.data_conf)



        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize_module(module = self.data_module, split='test')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.trainset,
            batch_size= self.hparams.batch_size,
            pin_memory=True,
            shuffle=True)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valset,
            batch_size= self.hparams.batch_size,
            pin_memory=True,
            shuffle=False)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.testset,
            batch_size= self.hparams.batch_size,
            pin_memory=True,
            shuffle=False)
        return test_loader
    
    def instancialize_module(self, module, **other_args):
        class_args =  list(inspect.signature(module.__init__).parameters)[1:]
        inkeys = other_args.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = other_args[arg]
        args1.update(other_args)
        return module(**args1)

    def init_data_module(self, name, **other_args):
        return getattr(importlib.import_module('data.framediff.dataset'), f'{name}_Dataset')