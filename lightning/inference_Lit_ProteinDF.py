import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="inference")
def run(conf: DictConfig) -> None:
    myconf = conf
    pass


if __name__ == '__main__':
    run()