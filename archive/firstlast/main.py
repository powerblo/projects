import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from environment import sphere_env
from tqdm.auto import tqdm
import numpy as np
from scipy.constants import pi

# path = './flpass/results/' # Use this for first passage
path = './flpass/results_alt/' # Use this for last passage

# System configuration
max_iteration = 250000000 # ~ 3 min for 2.5E10
num_bin = 1000 # Always even
num_agents = 50000000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pos = 0 # Initial conducting sphere position
rad = 5 # Conducting sphere radius
rad_inf = 100 # Infinity sphere radius, R
incr = 0.25 # Increment of moving conducting sphere; d = pos + i * incr; d < rad
incr_count = 20 # pos + incr_count < rad & rad_inf - rad

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Config:
    def __init__(self, distributed):
        self.rank = None
        self.distributed = distributed
        if distributed:
            self.world_size = torch.cuda.device_count()
            assert self.world_size > 1, 'More than 1 GPU need to be accessible for parallel training.'
        else:
            self.world_size = 1

def run_iteration(rank, config:Config):
    print(f"Running parallel off-centered WOS on rank {rank}.")
    setup(rank, config.world_size)
    for i in range(incr_count):
        config.rank = rank
        current_iteration = 0
        current_terminated = 0
        current_infinity = 0
        distribution = torch.zeros(size=(num_bin,), dtype=torch.float64, device=rank)
        env = sphere_env(pos=pos+i*incr, rad=rad, rad_inf=rad_inf, num_agent=num_agents, device=rank)
        range1 = range(max_iteration)
        pbar1 = tqdm(range1, desc='Simulation Progress '+str(i), total=max_iteration, leave = True, position=0, colour='blue')
        while current_terminated <= max_iteration:
            termination_num, infinity_num, terminal_z = env.step()
            terminal_z = terminal_z/rad # Scale to [-1,1]
            distribution += count_distribution(terminal_z)/bin_area(rank)
            num_resetted = termination_num + infinity_num
            pbar1.update(termination_num)
            current_iteration += num_resetted
            current_terminated += termination_num
            current_infinity += infinity_num
            prob_hit = current_terminated/current_iteration
            pbar1.set_postfix({"Prob_hit":f"{prob_hit:.4f}"})
        print(f"Finished parallel off-centered WOS on rank {rank}.")
        torch.save(distribution, path+'distribution_rank_'+ str(rank) + '_' + str(i) + '.pth')
    cleanup()

def count_distribution(terminal_z):
    temp = 0.5*(torch.clamp(terminal_z, min=-1, max=1) + 1) # Clamp between -1, 1, and then offset and scale to be [0,1].
    temp = torch.floor(num_bin * temp).to(torch.int) # Multiply by number of bins. 0 corresponds to z = -rad, num_bin corresponds to z = rad.
    count = torch.bincount(temp, minlength=num_bin)
    return count

def bin_area(device):
    '''
    Computes the area of each bins.
    '''
    index = torch.arange(num_bin).to(device)
    stepsize = 2*rad/num_bin
    bin_left = index*stepsize - rad
    bin_right = bin_left + stepsize
    out = 2 * pi * rad * (stepsize)
    return out

if __name__ == '__main__':
    use_distributed_training = True
    config = Config(use_distributed_training)
    mp.spawn(run_iteration, args=(config,), nprocs=config.world_size, join=True)