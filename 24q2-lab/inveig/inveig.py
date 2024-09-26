
import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
import itertools
from tqdm.auto import tqdm

def seq_mlp(init, mlp, fin, act):
    modules = [nn.Linear(init, mlp[0]), act]
    for i in range(len(mlp) - 1):
        modules.append(nn.Linear(mlp[i], mlp[i+1]))
        modules.append(act)

    modules.append(nn.Linear(mlp[-1], fin)) #self.spl for spline

    return modules

# |%%--%%| <FCwfdn1gJk|1lfI2QTNfq>

class EvalEig(nn.Module):
    def __init__(self, eval_para):
        super().__init__()   
        self.ln = eval_para['l_max']
        self.pw = eval_para['pw']
        self.batch_dim = eval_para['batch_dim']
        self.coeff_max = eval_para['coeff_max']

    def set_rdsc(self, rm, rn):
        self.rn = rn
        self.rm = rm
        self.r_dsc = torch.linspace(rm/rn, rm, rn)
        # batch x L x RxR
        self.l_dsc = torch.arange(0,self.ln+1, dtype = torch.int).view(1,-1,1,1)
        
        # R x pw
        r_dsc_adj = self.r_dsc.view(-1,1).expand(-1,self.pw)
        r_dsc_pc = r_dsc_adj * 1/torch.arange(1,self.pw+1).view(1,-1) # factorial factor for coefficients
        r_dsc_cp = torch.cumprod(r_dsc_pc, dim = 1)
        self.r_dsc_pw = torch.cat((1/self.r_dsc.view(-1,1), r_dsc_cp), dim = 1).T
    
    # set depends on range and distribution of values for coefficients
    def taylor_tr(self, para_ptl, ptl_form):
        coeffs_tr = torch.rand(self.batch_dim, self.pw+1)*para_ptl

        if ptl_form == "coulomb":
            coeffs_tr[:,0] = -1 + coeffs_tr[:,0]/10
            coeffs_tr[:,1:] = torch.tensor([0])
        elif ptl_form == "coulomb_like":
            coeffs_tr[:,0] = -1 + coeffs_tr[:,0]/10
            coeffs_tr[:,1:] = coeffs_tr[:,1:]*self.coeff_max
        elif ptl_form == "sho":
            coeffs_tr[:,0:] = torch.tensor([0])
            coeffs_tr[:,2] = 1
        elif ptl_form == "zero":
            coeffs_tr[:,0:] = torch.tensor([0])

        ptl_tr = coeffs_tr @ self.r_dsc_pw
        # batch x R
        return para_ptl*ptl_tr
    
    def fixed_tr(self, para_ptl, ptl_form):
        if ptl_form == "coulomb":
            coeffs_tr = torch.rand(self.batch_dim, 1)*para_ptl
            return -coeffs_tr/self.r_dsc.view(1,-1)
    
    def dsc_eigs(self, ptl):
        dsc_lap = (-2*torch.eye(self.rn) + torch.diag(torch.ones(self.rn-1),1) + torch.diag(torch.ones(self.rn-1),-1))/(2*(self.rm/self.rn)**2)
        dsc_lap = dsc_lap.view(1, 1, self.rn, self.rn)
        dsc_ptl = torch.diag_embed(ptl).view(self.batch_dim, 1, self.rn, self.rn)

        #dsc_lap = (-2*torch.eye(2*self.rn+1) + torch.diag(torch.ones(2*self.rn),1) + torch.diag(torch.ones(2*self.rn),-1))#/(2*(self.rm/self.rn)**2)
        #dsc_lap = dsc_lap.view(1, 1, 2*self.rn+1, 2*self.rn+1)
        #ptl_sym = torch.cat((torch.flip(ptl, dims = (1,)), torch.tensor([[1e-3]]), ptl), dim = 1)
        #dsc_ptl = torch.diag_embed(ptl_sym).view(self.batch_dim, 1, 2*self.rn+1, 2*self.rn+1)
        #r_dsc_sym = torch.cat((torch.flip(self.r_dsc, dims = (0,)), torch.tensor([1e-3]), self.r_dsc), dim = 0)

        #dsc_eff = self.l_dsc*(self.l_dsc+1)*torch.diag(1/r_dsc_sym**2).view(1, 1, 2*self.rn+1, 2*self.rn+1)
        dsc_eff = self.l_dsc*(self.l_dsc+1)*torch.diag(1/self.r_dsc**2).view(1, 1, self.rn, self.rn)
        dsc_hmt = -1*dsc_lap + dsc_ptl + 1*dsc_eff # fix coefficient hba^2/2m = 1

        evl, _ = torch.linalg.eigh(dsc_hmt)
        
        cutoff = evl.shape[2]
        for i in range(evl.shape[1]):
            cutoff = min(cutoff, evl[:,i][evl[:,i]<0].shape[0])
        cutoff = max(cutoff - 5, 5)

        return evl[:,:,:cutoff]
    


# |%%--%%| <1lfI2QTNfq|jtBYyAi6g9>

eval_para = {
        # evaluation model
        #'r_infty' : 1, # keep as <1 for numerical stability of taylor series
        #'r_dsc' : 1000, # computation time vs accuracy; maintain delta r ~ 0
        'l_max' : 2, # maximum l_max to evaluate radial schrodinger upto
        
        # potential specifics
        #'para_1' : 1e-3, # scales energy; ensure r_infty * para_1 < 1e-3 (roughly V(r_infty) << 1)

        # training data specifics
        #'ptl_form' : 'sho', # coulomb, coulomb_like
        'pw' : 20, # consider 1/r and up to r^n
        'coeff_max' : 100,

        # model specifics
        'precision' : 64, # 32 or 64 bit
        'batch_dim' : 50,
        }

model_para = {
        # model
        'mlp' : [100, 100],

        # training
        'epoch' : 1000,
        'lr' : 1e-2,

        # loss regularisation
        'reg1' : 1e-1, # smoothness
        
        }

eval = EvalEig(eval_para)

# |%%--%%| <jtBYyAi6g9|5FybpCh0Lz>

eval_grid = [[1000, 2000], [600], [2]] # rm, rn, para_1
for midx in itertools.product(*eval_grid):
    eval.set_rdsc(midx[0], midx[1])
    ptl_tr = eval.fixed_tr(midx[2], "coulomb")
    evl_tr = eval.dsc_eigs(ptl_tr)
    factor = torch.mean(1/(evl_tr/evl_tr[:,:,0].view(evl_tr.shape[0],evl_tr.shape[1],1).expand(-1, evl_tr.shape[1], evl_tr.shape[2])), dim = 0)
    print(midx, nn.L1Loss()(factor[0,:5],torch.arange(1,6)**2), evl_tr[0,0,0])

# |%%--%%| <5FybpCh0Lz|B3yip8y0tv>


