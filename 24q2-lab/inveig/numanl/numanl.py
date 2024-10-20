
import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchdiffeq import odeint

# |%%--%%| <4EsaucioNG|OgpsVD15m6>

class EvalEig(nn.Module):
    def __init__(self, r, evl_pair_init, step_size, p0):
        # evl_guess : evl_min, evl_max, evl_n
        super().__init__()
        self.r_dsc = torch.linspace(r[0], r[1], r[2])
        self.evl_pair_init = evl_pair_init
        
        self.step_size, self.p0 = step_size, p0
    
    def evl_anl(self):
        integ = torch.arange(1,self.evl_n+1)
        clm_evl = (-1/(4*integ**2))*self.p0**2#*1/(4*np.pi)**2
        return clm_evl

    def ode_eq(self, r, init_rad):
        rad, rad_d = init_rad[:,0], init_rad[:,1]
        ptl_rs = -self.p0/r * torch.ones(self.evl_n)
        rad_dd = (ptl_rs - self.evl_guess) * rad
        derivs = torch.stack([rad_d, rad_dd.squeeze(-1)],dim=-1)
        return derivs

    def evl_node(self, evl_guess):
        self.evl_guess, self.evl_n = evl_guess, evl_guess.shape[0]
        init_rad_rs = torch.stack([torch.ones(self.evl_n)*0,
                                   torch.ones(self.evl_n)],dim=-1)
        
        clm_rad = odeint(self.ode_eq, init_rad_rs, self.r_dsc, 
                        method = 'rk4', options=dict(step_size = self.step_size))
                        #method = 'dopri5')
        clm_radinf = clm_rad[-1,:,0]
           
        clm_vals = [[],[]]
        for i in range(clm_radinf.shape[0]-1):
            if clm_radinf[i]*clm_radinf[i+1]<0:
                clm_vals[0].append(self.evl_guess[i].item())
                clm_vals[1].append(i)
        
        return clm_rad[:,:,0].transpose(0,1), clm_vals

    def evl_alg(self):
        evl_guess = torch.linspace(self.evl_pair_init[0]*self.p0**2,0,self.evl_pair_init[1])
        evl_del = evl_guess[1] - evl_guess[0]
        for i in range(10):
            _, evl_hits = self.evl_node(evl_guess)
            print(evl_hits[0][0]/torch.tensor(evl_hits[0]))
            evl_guess_stack = []
            for i in range(len(evl_hits[1])):
                evl_guess_stack.append(torch.linspace(evl_hits[0][i] - evl_del/2**i, evl_hits[0][i] + evl_del/2**i, int(self.evl_pair_init[1]/2)))
            evl_guess_stack.append(torch.linspace(evl_hits[0][len(evl_hits[1])-1],0,int(self.evl_pair_init[1]/2)))
            evl_guess = torch.stack(evl_guess_stack).view(-1)

#|%%--%%| <OgpsVD15m6|uv5alQ53fx>

# heuristics : sho : lower rm preferred; clm : higher rm preferred
model = EvalEig(r = [1e-1, 2000, 2000], # rmin, rmax, rn; points at where u(r) is evaluated
                evl_pair_init = [-0.3, 1000], # e_ground, en; guess for ground energy WHEN p0=1 (-0.25 for coulomb), no. of points where u_infty(E) is evaluated
                step_size = 0.1, # fixed step size in rk4
                p0 = 0.1) # constant parameter of potential

model.evl_alg() # rad[E,r]

#|%%--%%| <uv5alQ53fx|yr4s4Vp0pC>

def eigen_rad_plot(rad, evl):
    #plt.figure()
    for i in range(4,len(evl[1])):
        plt.plot(range(rad.shape[1]), rad[evl[1,i].to(int)]) 

eigen_rad_plot(rad, evl)

#|%%--%%| <yr4s4Vp0pC|PX0ESPYiHy>

#plt.figure()
#plt.plot(range(rad[i].shape[0]),rad[i])
#plt.plot(range(rad[:,-1].shape[0]),torch.log(torch.abs(rad[:,-1])))
#plt.plot(range(evl.shape[0]), evl)

