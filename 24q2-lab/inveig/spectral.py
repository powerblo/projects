
import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from torch.nn.modules.batchnorm import init
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
    def __init__(self, para):
        super().__init__()
        self.rn = para['rn']
        self.rm = para['rm']
    
    def fixed_tr(self, para_ptl, ptl_form):
        coeffs_tr = torch.ones(self.batch_dim, 1)
        r_dsc_d = self.r_dsc.view(1,-1)
        if ptl_form == "coulomb":
            #coeffs_tr = 1 + torch.rand(self.batch_dim, 1)/10
            return -coeffs_tr*para_ptl/r_dsc_d
        elif ptl_form == "yukawa":
            return coeffs_tr*para_ptl*(-1/r_dsc_d + torch.exp(-r_dsc_d/40)*r_dsc_d/120)

    def set_evl(self, evl):
        self.evl = evl
        
        ptl_modules = seq_mlp(init = 1, mlp = para['mlp'], fin = 1, act = nn.ReLU())
        self.ptl_mlp = nn.Sequential(*ptl_modules)
        rad_modules = seq_mlp(init = 1, mlp = para['mlp'], fin = evl.shape[0], act = nn.Tanh())
        self.rad_mlp = nn.Sequential(*rad_modules)        

    def grad_nn(self, x, y):
        grad_list = []
        for i in range(y.shape[0]):
            grad = torch.autograd.grad(
                outputs = y[i], inputs = x, grad_outputs = torch.ones_like(y[i]), 
                retain_graph = True, create_graph = True)[0]
            grad_list.append(grad)
        grad = torch.stack(grad_list)
        return grad

    def spectral_err(self, rad, ptl, evl):
        r_dsc = torch.linspace(self.rm/self.rn, self.rm, self.rn, requires_grad=True)
        self.r_dsc = r_dsc.detach()
        r_rs = r_dsc.unsqueeze(0).expand(evl.shape[0],-1)

        evl_rs = evl.view(-1,1)
        ptl_rs = ptl(r_dsc.view(-1,1)).view(1,-1)
        self.ptl_rs = ptl_rs.detach()
        rad_rs = rad(r_dsc.view(-1,1)).transpose(0,1)
        #print("u(r) : ", rad_rs.shape, rad_rs)

        rad_d = self.grad_nn(r_dsc, rad_rs)
        #print("u'(r) : ", rad_d.shape, rad_d)

        rad_dd = self.grad_nn(r_dsc, rad_d)
        #print("u''(r) : ", rad_dd.shape, rad_dd)
        
        #plt.figure()
        #plt.plot(r_dsc.detach(), rad_rs[0].detach())
        #plt.plot(r_dsc.detach(), rad_d[0].detach())
        #plt.plot(r_dsc.detach(), rad_dd[0].detach())
        #plt.show()

        # rad output : evl_N x self.rn
        error_mtr = -rad_dd + ptl_rs * rad_rs - evl_rs * rad_rs

        return error_mtr

    def forward(self):
        error_mtr = self.spectral_err(self.rad_mlp, self.ptl_mlp, self.evl)
        error = torch.sum(torch.abs(error_mtr))

        return error

# |%%--%%| <Nmb5q8f54j|xr3NyHiOFF>

para = {
        'rm' : 1,
        'rn' : 1000,

        # model
        'mlp' : [100,100],

        # training
        'epoch' : 1000,
        'lr' : 1e-2,

        # loss regularisation
        'reg1' : 1e-2,
        'reg2' : 1e-2,
}

model = EvalEig(para)

#|%%--%%| <xr3NyHiOFF|WRUrqeNtUf>

evl_tr = -torch.arange(1,10).to(torch.float32)**(-2)
model.set_evl(evl_tr)

# |%%--%%| <WRUrqeNtUf|dyJzB2ZAcV>

#model.load_state_dict(torch.load('1.pth'))

optimiser = torch.optim.Adam(model.parameters(), lr = para['lr'])
epochs = para['epoch']
pbar = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')
loss_list = [[],[]]

for e in range(epochs):
    #with torch.autograd.detect_anomaly():
    loss = model()
    #print(loss.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    #pbar.update()
    print(loss.item())

#|%%--%%| <dyJzB2ZAcV|6rxWjy8s0O>

print(loss.item())
plt.figure()
plt.plot(model.r_dsc, model.ptl_rs[0])
plt.show()
