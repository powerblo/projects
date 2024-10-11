
import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm

def seq_mlp(init, mlp, fin, act):
    modules = [nn.Linear(init, mlp[0]), act]
    for i in range(len(mlp) - 1):
        modules.append(nn.Linear(mlp[i], mlp[i+1]))
        modules.append(act)

    modules.append(nn.Linear(mlp[-1], fin)) #self.spl for spline

    return modules

# |%%--%%| <4EsaucioNG|OgpsVD15m6>

class EvalEig(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.rn = para['rn']
        self.rm = para['rm']
        self.r_dsc = torch.linspace(self.rm/self.rn, self.rm, self.rn, requires_grad=True)
    
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
        evl_rs = evl.view(-1,1)
        ptl_rs = ptl(self.r_dsc.view(-1,1)).view(1,-1)
        rad_rs = rad(self.r_dsc.view(-1,1)).transpose(0,1)

        # boundary conditions on rad_rs
        #rad_rs = rad_rs * 

        rad_norm = rad_rs / torch.sqrt(torch.sum(rad_rs**2, dim=1).view(-1,1)*self.rm/self.rn)
        #print("u(r) : ", rad_rs.shape, rad_rs)

        rad_d = self.grad_nn(self.r_dsc, rad_norm)
        #print("u'(r) : ", rad_d.shape, rad_d)

        rad_dd = self.grad_nn(self.r_dsc, rad_d)
        #print("u''(r) : ", rad_dd.shape, rad_dd)
        
        #plt.figure()
        #plt.plot(r_dsc.detach(), rad_rs[0].detach())
        #plt.plot(r_dsc.detach(), rad_d[0].detach())
        #plt.plot(r_dsc.detach(), rad_dd[0].detach())
        #plt.show()

        # rad output : evl_N x self.rn
        error_mtr = -rad_dd + ptl_rs * rad_norm - evl_rs * rad_norm

        return ptl_rs, rad_norm, error_mtr

    def forward(self):
        ptl_md, rad_md, error_mtr = self.spectral_err(self.rad_mlp, self.ptl_mlp, self.evl)
        error = torch.sqrt(torch.sum(error_mtr**2))

        return ptl_md, rad_md, error

# |%%--%%| <OgpsVD15m6|P85EjLGoIH>

para = {
        'rm' : 100,
        'rn' : 1000,

        # model
        'mlp' : [10,10,10],

        # training
        'epoch' : 100,
        'lr' : 1e-4,

        # loss regularisation
        'reg0' : 1,
        'reg1' : 1e2,
        'reg2' : 1e4,
}

model = EvalEig(para)

evl_tr = -torch.arange(1,10).to(torch.float32)**(-2)
model.set_evl(evl_tr)

# |%%--%%| <P85EjLGoIH|EyX0aMsDTz>

optimiser = torch.optim.Adam(model.parameters(), lr = para['lr'])
epochs = para['epoch']
pbar = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')
loss_list = [[],[],[]]

for e in range(epochs):
    #with torch.autograd.detect_anomaly():
    ptl_md, rad_md, loss0 = model()
    #print(loss.item())
    loss1 = torch.abs(ptl_md[0,-1])
    loss2 = torch.sqrt(torch.sum(rad_md[:,0]**2 + rad_md[:,-1]**2))
    loss = para['reg0']*loss0 + para['reg1']*loss1 + para['reg2']*loss2

    loss_list[0].append(loss0.item())
    loss_list[1].append(loss1.item())
    loss_list[2].append(loss2.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    pbar.update()
    #print(loss.item())

# |%%--%%| <EyX0aMsDTz|I83gApPhjy>

plt.figure()
plt.plot(model.r_dsc.detach(), ptl_md[0].detach())
plt.show()

# |%%--%%| <I83gApPhjy|HGfqpL7M7D>

plt.figure()
plt.plot(model.r_dsc.detach(), rad_md[0].detach())
plt.show()

# |%%--%%| <HGfqpL7M7D|9PeZd3M3YA>

plt.figure()
j = 0
print(loss0.item(),loss1.item(),loss2.item())
plt.plot(np.arange(j,epochs), loss_list[0][j:])
plt.plot(np.arange(j,epochs), loss_list[1][j:])
plt.plot(np.arange(j,epochs), loss_list[2][j:])

# |%%--%%| <9PeZd3M3YA|2yzR9m31Ac>


