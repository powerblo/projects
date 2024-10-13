import torch

class Config:
    def __init__(self, distributed, hp):
        self.rank = None
        self.distributed = distributed
        
        self.hp = hp

        if distributed:
            self.world_size = torch.cuda.device_count()
            assert self.world_size > 1, 'More than 1 GPU need to be accessible for parallel training.'
        else:
            self.world_size = 1
