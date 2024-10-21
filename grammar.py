import torch.distributed as dist
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("gloo", rank=1, world_size=1)

rank = dist.get_rank()
print(dist.get_world_size())