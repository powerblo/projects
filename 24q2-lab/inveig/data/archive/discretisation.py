import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm

#|%%--%%| <z5cNqQMy1y|dVKYTPv7jo>

class EvalEig(nn.Module):
    def __init__(self, eval_para):
        super().__init__()
        self.rn = eval_para['r_dsc']
        self.rm = eval_para['r_infty']        
        self.ln = eval_para['l_max']
        self.para = eval_para
        self.b_dim = eval_para['batch_dim']

        self.r_dsc = torch.linspace(self.rm/self.rn, self.rm, self.rn)
        # batch x L x RxR
        self.l_dsc = torch.arange(0,self.ln+1, dtype = torch.int).view(1,-1,1,1)
        
        # R x pw
        r_dsc_adj = self.r_dsc.view(-1,1).expand(-1,self.para['pw'])
        r_dsc_pc = r_dsc_adj # * 1/torch.arange(1,eval_para['pw]+1) # factorial factor for coefficients
        r_dsc_cp = torch.cumprod(r_dsc_pc, dim = 1)
        self.r_dsc_pw = torch.cat((1/self.r_dsc.view(-1,1), r_dsc_cp), dim = 1).T
    
    # set depends on range and distribution of values for coefficients
    def taylor_tr(self):
        coeffs_tr = torch.rand(self.para['batch_dim'], self.para['pw']+1)*self.para['para_1']
        coeffs_tr[:,0] = -torch.abs(coeffs_tr[:,0])

        if self.para['ptl_form'] == "coulomb":
            coeffs_tr[:,1:] = torch.tensor([0])
        elif self.para['ptl_form'] == "coulomb_like":
            coeffs_tr[:,1:] = coeffs_tr[:,1:]*self.para['coeff_max']

        ptl_tr = coeffs_tr @ self.r_dsc_pw
        return self.para['para_1']*ptl_tr
    
    def dsc_eigs(self, ptl):
        dsc_lap = (-2*torch.eye(self.rn) + torch.diag(torch.ones(self.rn-1),1) + torch.diag(torch.ones(self.rn-1),-1))/self.rn**2
        dsc_lap = dsc_lap.view(1, 1, self.rn, self.rn)
        dsc_ptl = torch.diag_embed(ptl).view(self.b_dim, 1, self.rn, self.rn)
        dsc_eff = self.l_dsc*(self.l_dsc+1)*torch.diag(1/self.r_dsc**2).view(1, 1, self.rn, self.rn)
        
        dsc_hmt = (-self.para['para_0']*dsc_lap + dsc_ptl + self.para['para_0']*dsc_eff)

        evl, _ = torch.linalg.eigh(dsc_hmt)
        
        cutoff = evl.shape[2]
        for i in range(evl.shape[1]):
            cutoff = min(cutoff, evl[:,i][evl[:,i]<0].shape[0])
        cutoff -= 5

        return evl[:,:,:cutoff]


#|%%--%%| <dVKYTPv7jo|YomJKi1WzO>

eval_para = {
        # evaluation model
        'r_infty' : 1, # keep as <1 for numerical stability of taylor series
        'r_dsc' : 1000, # computation time vs accuracy; maintain delta r ~ 0
        'l_max' : 2, # maximum l_max to evaluate radial schrodinger upto
        
        # potential specifics
        'para_0' : 1, # hbar^2/2m
        'para_1' : 1e-3, # scales energy; ensure r_infty * para_1 < 1e-3 (roughly V(r_infty) << 1)

        # training data specifics
        'ptl_form' : 'coulomb', # coulomb, coulomb_like
        'pw' : 20, # consider 1/r and up to r^n
        'coeff_max' : 100,

        # model specifics
        'precision' : 64, # 32 or 64 bit
        'batch_dim' : 1,
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

#|%%--%%| <YomJKi1WzO|T7znv60kBQ>

ptl_tr = eval.taylor_tr()

#plt.figure()
#plt.plot(eval.r_dsc, ptl_tr[0].squeeze(0).detach(), label='model', color = 'blue')
#plt.legend()
#plt.show()

#|%%--%%| <T7znv60kBQ|qoqVgpAehK>

evl_tr = eval.dsc_eigs(ptl_tr)

print(evl_tr)
print(evl_tr[0,0,0]/evl_tr)
