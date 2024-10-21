import hydra
from omegaconf import DictConfig
import os
import pickle

import logging
from time import time
import os
import hydra
import logging
import pickle
import struct
from omegaconf import OmegaConf,open_dict

LOG = logging.getLogger(__name__)
@hydra.main(version_base=None, config_path="config", config_name="inference")
def run(conf: DictConfig) -> None:
    myconf = conf
    LOG.info(f'hydra.utils.get_original_cwd(): {hydra.utils.get_original_cwd()}')

    with open(f'log.pkl', 'wb') as f:
        pickle.dump(
            {'all_UP': 1}, f)


if __name__ == '__main__':
    run()