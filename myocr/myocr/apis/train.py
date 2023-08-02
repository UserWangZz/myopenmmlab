import numpy as np
import torch
import torch.distributed as dist

from mycv.runner import get_dist_info

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed. If the seed is None, it will be replaced by a
    random number, and then broadcasted to all processes.

    Args:
        seed (int, Optional): The seed.
        device (str): The device where the seed will be put on.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()

if __name__ == '__main__':
    seed = init_random_seed()
    print(seed)