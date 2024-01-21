import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm.auto import tqdm

# repeatN = 1000, batchN = 10000000, binningN = 1000
repeatN = 200 # number of walks before termination
targetN = 1e8 # target number of hits before termination -> only use one of the two!

batchN = 50000000   # number of points evaluated at once
binningN = 1000 # unit of splitting binning of angles
eps = 1e-5   # tolerance for collision with sphere

inf_r = 100
int_r = 5
int_d_start = 85 # 0
int_d_inc = 100

iterN = round((inf_r-int_d_start)/int_d_inc)

path = './results/'

def euclidean(points1, points2):
    return torch.sqrt(torch.sum((points1 - points2)**2, dim=1))

def clamp(h_points):
    temp = 0.5*(torch.clamp(h_points, min = -1+eps, max = 1-eps) + 1)
    temp = torch.floor(binningN * temp).to(torch.int)
    count = torch.bincount(temp, minlength = binningN)
    return count

class ensemble:
    def __init__(self, inf_r, int_r, int_d, device:None):
        self.device = device
        self.inf_r = inf_r
        self.int_r = int_r
        self.int_d = int_d
        self.points = self.random_spherical(self.uniform_vector(self.inf_r))
        self.int_centre_tensor = self.uniform_tensor([0,0,self.int_d], batchN)

    def uniform_vector(self, radius):
        return torch.ones(size = (batchN,), dtype = torch.float64, device = self.device) * radius
    
    def uniform_tensor(self, vector, size):
        return torch.tensor(vector, device = self.device).repeat(1,size,1).squeeze(0)

    def random_tensor(self, size):
        return torch.rand(size = (size,), dtype = torch.float64, device = self.device)

    def random_spherical(self, radii:torch.float64):
        theta = torch.acos(2 * self.random_tensor(radii.size(dim = 0)) - 1)
        phi = 2 * np.pi * self.random_tensor(radii.size(dim = 0))
 
        x = radii * torch.sin(theta) * torch.cos(phi)
        y = radii * torch.sin(theta) * torch.sin(phi)
        z = radii * torch.cos(theta)
        return torch.stack((x, y, z), dim=1)
    
    def walk_on_sphere(self):
        walk_dist = euclidean(self.points, self.int_centre_tensor) - self.int_r
        
        self.points += self.random_spherical(walk_dist)
    
    def infinity_points(self):
        norms = torch.norm(self.points, dim=1)
        inf_prob = 1 - self.inf_r / norms
        mask_inf = (self.random_tensor(batchN) < inf_prob).squeeze(0)

        self.points[mask_inf] = self.random_spherical(self.uniform_vector(self.inf_r)[mask_inf])
    
    def hit_angle(self, int_points):
        int_displ = int_points - self.uniform_tensor([0,0,self.int_d], int_points.size(dim = 0))

        z_pos = torch.div(int_displ[:,2], torch.norm(int_displ, dim=1))

        return z_pos

    def hit_points(self):
        int_dist = euclidean(self.points, self.int_centre_tensor)
        mask_int = (int_dist < self.uniform_vector(self.int_r + eps)).squeeze(0)

        h_points = self.points[mask_int]

        self.points[mask_int] = self.random_spherical(self.uniform_vector(self.inf_r)[mask_int])

        return self.hit_angle(h_points)

    def step(self):
        self.walk_on_sphere()
        # self.infinity_points()
        return self.hit_points()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

class Config:
    def __init__(self, distributed):
        self.rank = None
        self.distributed = distributed
        if distributed:
            self.world_size = torch.cuda.device_count()
            assert self.world_size > 1, 'More than 1 GPU need to be accessible for parallel training.'
        else:
            self.world_size = 1

def simulate(rank, config:Config):
    print(f"Running parallel walk-on-spheres on rank {rank}.")
    config.rank = rank

    setup(rank, config.world_size)

    pbar1 = tqdm(range(repeatN), desc='Simulation Progress', total=targetN * iterN, leave = True, position=0, colour='blue')

    for int_ind in range(iterN):
        int_d = int_ind * int_d_inc + int_d_start

        ens = ensemble(inf_r, int_r, int_d, device = rank)
        hit_stat = torch.zeros(size = (binningN,), dtype = torch.int, device = rank)

        hit_c = 0
        index = 0

        # while index <= targetN:
        while index <= repeatN:
            hit_batch = ens.step()
            hit_s = hit_batch.size(dim = 0)
            hit_stat += clamp(hit_batch)

            # index += hit_s
            index += 1

            pbar1.update(hit_s)
    
        torch.save(hit_stat, path+'dist_'+str(inf_r)+'_'+str(int_r)+'_'+str(int_d)+'_rank_'+str(rank)+'.pth')

        if single:
            break

    dist.destroy_process_group()

if __name__ == "__main__":
    use_distributed_training = True
    config = Config(use_distributed_training)
    mp.spawn(simulate, args=(config,), nprocs=config.world_size, join=True)