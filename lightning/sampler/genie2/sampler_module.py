from tqdm import tqdm

from lightning.sampler.genie2.unconditional_runner import UnconditionalRunner
from lightning.sampler.genie2.scaffold_runner import ScaffoldRunner

from lightning.sampler.genie2.multiprocessor import MultiProcessor

from omegaconf import DictConfig, OmegaConf
import torch
import logging

class genie2_Sampler:
    def __init__(self, conf: DictConfig):
        self.log = logging.getLogger(__name__)
        self.inference_ready = False
        self.conf = conf
        self.infer_conf = conf.inference
        self.output_dir = self.infer_conf.output_dir
        self.num_devices = torch.cuda.device_count()
        self.log.info(f'Inference on {self.num_devices} GPUs.')

        # self.lightning_model = genie2_Lightning_Model.load_from_checkpoint(self.inference_ckpt)
        if self.infer_conf.task_type == "unconditional":
            self.runner = UnconditionalRunner()
        elif self.infer_conf.task_type == "scaffold":
            self.runner = ScaffoldRunner()

    def run_sampling(self):
        self.runner.run(self.infer_conf, self.num_devices)







