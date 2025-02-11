import torch

def get_max_memory_allocated():
    return torch.cuda.max_memory_allocated(0) / 1024 ** 2
