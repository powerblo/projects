
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

# |%%--%%| <oLmbyD3rxN|Nmb5q8f54j>

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
        coeffs_tr = torch.ones(self.batch_dim, 1)
        r_dsc_d = self.r_dsc.view(1,-1)
        if ptl_form == "coulomb":
            #coeffs_tr = 1 + torch.rand(self.batch_dim, 1)/10
            return -coeffs_tr*para_ptl/r_dsc_d
        elif ptl_form == "yukawa":
            return coeffs_tr*para_ptl*(-1/r_dsc_d + torch.exp(-r_dsc_d/40)*r_dsc_d/120)
        elif ptl_form == "monotonic":
            return coeffs_tr*para_ptl*(
                -0.5 - 1/r_dsc_d + 0.5/(1+torch.exp(-r_dsc_d/15+40))
            )
    
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

        return evl
    
    def evl_cutoff(self, evl):
        cutoff = evl.shape[2]
        for i in range(evl.shape[1]):
            cutoff = min(cutoff, evl[:,i][evl[:,i]<0].shape[0])
        cutoff = max(cutoff - 5, 5)

        self.cutoff = cutoff
        return cutoff
    
    def evl_scl(self, evl):
        evl_scl = (evl/evl[:,:,0].view(evl.shape[0],evl.shape[1],1).expand(-1, evl.shape[1], evl.shape[2]))[:,:,1:]
        return evl_scl
    
    def forward(self, rm, rn, para_ptl, ptl_form):
        self.set_rdsc(rm, rn)
        ptl_tr = self.fixed_tr(para_ptl, ptl_form)
        evl_tr = self.dsc_eigs(ptl_tr) # eval true eigenvalues
        evl_tr_cut = evl_tr[:,:,:self.evl_cutoff(evl_tr)]
        #evl_tr_scl = self.evl_scl(evl_tr_cut)

        return ptl_tr, evl_tr_cut #evl_tr_scl

class InvEig(EvalEig):
    def __init__(self, eval_para, model_para):
        super().__init__(eval_para)

    def set_rdsc(self, rm, rn, cutoff):
        self.rn = rn
        self.rm = rm
        self.r_dsc = torch.linspace(rm/rn, rm, rn)
        # batch x L x RxR
        self.l_dsc = torch.arange(0,self.ln+1, dtype = torch.int).view(1,-1,1,1)
        self.cutoff_md = cutoff

        # initialise model
        #self.ptl = nn.Parameter(torch.rand(self.batch_dim, self.rn-1)) # random parameters
        #modules = seq_mlp(init = self.rn, mlp = model_para['mlp'], fin = self.rn-1, act = nn.ReLU())
        modules = seq_mlp(init = 1, mlp = model_para['mlp'], fin = 1, act = nn.ReLU())
        self.mlp = nn.Sequential(*modules)
        self.expn = nn.Parameter(torch.tensor([1.]))

    def forward(self, energy):
        # obtain potential via model
        #ptl_md = torch.cat((torch.tensor([-self.rn/self.rm]).view(1,-1).expand(self.batch_dim,-1),self.ptl),dim=1) # random parameters
        #ptl_md = torch.cat((torch.tensor([-self.rn/self.rm]).view(1,-1).expand(self.batch_dim,-1),self.mlp(self.r_dsc).view(1,-1).expand(self.batch_dim,-1)),dim=1) # random parameters
        #ptl_md = self.mlp(self.r_dsc.view(-1,1)).view(1,-1).expand(self.batch_dim,-1)*(1/self.r_dsc**self.expn)
        ptl_md = self.mlp(self.r_dsc.view(-1,1)).view(1,-1).expand(self.batch_dim,-1)*(1/self.r_dsc)

        evl_md = self.dsc_eigs(ptl_md)
        evl_md_cut = evl_md[:,:,:self.cutoff_md]
        #evl_md_scl = self.evl_scl(evl_md_cut)

        return ptl_md, evl_md_cut #evl_md_scl


# |%%--%%| <Nmb5q8f54j|xr3NyHiOFF>

eval_para = {
        # evaluation model
        #'r_infty' : 1, # keep as <1 for numerical stability of taylor series
        #'rn' : 1000, # computation time vs accuracy; maintain delta r ~ 0
        'l_max' : 2, # maximum l_max to evaluate radial schrodinger upto
        
        # potential specifics
        #'para_1' : 1e-3, # scales energy; ensure r_infty * para_1 < 1e-3 (roughly V(r_infty) << 1)

        # training data specifics
        #'ptl_form' : 'sho', # coulomb, coulomb_like
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
        'epoch' : 5000,
        'lr' : 1e-2,

        # loss regularisation
        'reg1' : 1e-1, # V(0) sign
        'reg2' : 1, # V -> 0 as r -> infty
        
        }

eval = EvalEig(eval_para)

# |%%--%%| <xr3NyHiOFF|mZOqyJgKBu>

#eval_grid = [[800], \
#    [10000], \
#        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] # rm, rn, para_1
#for midx in itertools.product(*eval_grid):
#for midx in zip(*eval_grid):
    #eval.set_rdsc(midx[0], midx[1])
    #ptl_tr = eval.fixed_tr(midx[2], "coulomb")
    #evl_scl_tr = eval.dsc_eigs(ptl_tr)
    #evl_tr = evl_scl_tr[:,:,:eval.evl_cutoff(evl_scl_tr)]
#    ptl_tr, evl_tr = eval(midx[0], midx[1], midx[2], "coulomb")
#    factor = torch.mean(1/evl_tr, dim = 0)
#    print(factor[0])
#    print(midx, nn.L1Loss()(factor[0],torch.arange(1,factor.shape[1]+1)**2), evl_tr[0,0,0])

# |%%--%%| <mZOqyJgKBu|Lf0ZAem6kd>

ptl_tr, evl_tr = eval(800, 1000, 1, "monotonic")

print(evl_tr)
print(ptl_tr.shape)

# |%%--%%| <Lf0ZAem6kd|dyJzB2ZAcV>

model = InvEig(eval_para, model_para)
model.set_rdsc(800, 1000, eval.cutoff)
model.load_state_dict(torch.load('1.pth'))

optimiser = torch.optim.Adam(model.parameters(), lr = model_para['lr'])
epochs = model_para['epoch']
pbar = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')
loss_list = [[],[]]

for e in range(epochs):
    #ptl_md, evl_md = model(hp['true'])
    #with torch.autograd.detect_anomaly():
    
    ptl_md, evl_md = model(evl_tr)
    if e == 0:
        ptl_init = ptl_md

    loss0 = nn.L1Loss()(evl_tr, evl_md)
    #loss0 = nn.L1Loss()(ptl_md, ptl_tr)

    #loss1 = torch.sum((ptl_md[:,1:]-ptl_md[:,:-1])**2)
    loss1 = torch.sum(nn.functional.relu(ptl_md[:,0]))
    loss2 = torch.sum(torch.abs(ptl_md[:,-1]))

    loss_list[0].append(loss0.item())
    loss_list[1].append(loss1.item())

    loss = loss0 + model_para['reg1']*loss1 \
        + model_para['reg2']*loss2

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    pbar.update()

# |%%--%%| <dyJzB2ZAcV|e9rRdHJzmQ>

torch.save(model.state_dict(), f"{eval_para['batch_dim']}.pth")


# |%%--%%| <e9rRdHJzmQ|5FwMYLzmY2>

plt.figure()
i = 0
plt.plot(eval.r_dsc, ptl_tr[i], label='true', color = 'red')
plt.plot(eval.r_dsc, ptl_init[i].squeeze(0).detach(), label='model_i', color = 'green')
plt.plot(eval.r_dsc, ptl_md[i].squeeze(0).detach(), label='model', color = 'blue')
plt.legend()
plt.annotate(f"E L1Loss : {loss:.3f}", xy = (0.75, 0.3), xycoords="axes fraction")
plt.annotate(f"V L1Loss : {nn.L1Loss()(ptl_tr[i], ptl_md[i]).item():.2f}", xy = (0.75, 0.25), xycoords="axes fraction")
plt.savefig(f"{eval_para['batch_dim']}_{eval_para['pw']}_{i}ptl.png")
print(evl_md[i])
print(evl_tr[i])

# |%%--%%| <5FwMYLzmY2|22QNVzAeWy>

plt.figure()
j = 0
plt.plot(range(epochs-j), loss_list[0][j:])
#plt.plot(range(epochs-j), loss_list[1][j:])
plt.savefig(f"{eval_para['batch_dim']}_{eval_para['pw']}_loss.png")

# |%%--%%| <22QNVzAeWy|YxOzQ8BeME>
r"""°°°

°°°"""
# |%%--%%| <YxOzQ8BeME|DfnsZtwiD8>
r"""°°°

°°°"""
# |%%--%%| <DfnsZtwiD8|JWWC3eGrPQ>


