import os
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import logging
import pandas as pd
import shutil
from datetime import datetime
from biotite.sequence.io import fasta
import GPUtil
from typing import Optional

from lightning.sampler.framediff import utils as su
from preprocess.tools import utils as du
from preprocess.tools import residue_constants
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from evaluate.openfold.data import data_transforms
import esm
from lightning.model.framediff.lightning_model import framediff_Lightning_Model


class Sampler:
    def __init__(self, conf: DictConfig):
        self.conf = conf
        self.infer_conf = conf.inference
        self.diff_conf = self.infer_conf.diffustion
        self.sample_conf = self.infer_conf.samples

        self._rng = np.random.default_rng(self.infer_conf.seed)
        self.inference_ckpt = self.infer_conf.weights_path
        self.model = framediff_Lightning_Model.load_from_checkpoint(self.inference_ckpt)

    def inference_fn(self):
        pass



