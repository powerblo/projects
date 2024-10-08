import numpy as np
import matplotlib.pyplot as plt 
import torch, torch.nn as nn
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

# layer depth : N = 10
# vel discretisation : L = 251

tdis = 10
vdis = 251

# numerical constants 

gnum = 10
tinum = 0
tfnum = 4
mnum = 1
delt = (tfnum - tinum)/tdis
eps = 0.0001

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dragf = nn.Parameter(10 + torch.rand(vdis)*10)

    def ipol(self, v):
        floor = torch.abs(torch.floor(v).long())
        ceil = torch.abs(torch.ceil(v).long())
        v = v + eps
        return self.dragf[floor]*(torch.ceil(v) - v + eps) \
                + self.dragf[ceil]*(v - torch.floor(v) - eps)

    def forward(self, v):
        for _ in range(tdis):
            v = v - (gnum - self.ipol(v)/mnum)*delt
        return v 

# numerical evaluation

vinit_train = np.linspace(-250,0,51)
vi = torch.tensor(vinit_train)
veval = np.linspace(0,vdis-1,vdis)

def dragf_true(v):
    va = np.abs(v)
    return va*(300-va)/1000*(1+np.sin(va/20)/10+np.cos(va/40)/10)+(va/70)**2 

def eom(t, v):
    return -gnum + dragf_true(v)/mnum 

vfinal_train = [] 
for vinit in np.linspace(-250,0,51):
    sol = solve_ivp(eom, (tinum, tfnum), [vinit], method='RK45', t_eval = np.linspace(tinum, tfnum, tdis*100)) 
    vfinal_train.append(sol.y[0][-1])

# model training 
## hyperparas
epoch = 2000
subepochs = [100, 200]
eta = 0.1
reg = 0.03

model = Model()
y_model = model(vi)
vfinal_epoch0 = y_model.detach().numpy()
dragf_epoch0 = model.dragf[veval].detach().numpy()

optim = torch.optim.Adam(model.parameters(), lr = eta)

pbar1 = tqdm(range(epoch), desc='Training Progress', total=epoch, leave=True, position=0, colour='blue')

for i in range(epoch):
    y_model = model(vi)
    
    dragf = model.dragf
    loss = nn.L1Loss()(torch.tensor(vfinal_train), y_model) + dragf[0]**2 + reg*torch.sum((dragf[1:] - dragf[:-1]) ** 2)

    optim.zero_grad()
    loss.backward()
    optim.step()

    pbar1.update(1)

vfinal_epochf = model.forward(vi).detach().numpy()
dragf_epochf = model.dragf[veval].detach().numpy()

# vi-vf graph
plt.figure()
plt.plot(vinit_train, vfinal_train, marker = 'o', color = 'blue', label = 'true') # training data
plt.plot(vinit_train, vfinal_epoch0, marker = 'o', color = 'red', label = 'epoch 0')
plt.plot(vinit_train, vfinal_epochf, marker = 'o', color = 'orange', label = 'epoch f')

plt.xlabel('v_i')
plt.ylabel('v_f')
plt.legend()

plt.savefig('vi-vf.png')
plt.close()

# v-dragf graph
plt.figure()
plt.plot(veval, dragf_true(veval), color = 'blue', label = 'true') # true data
plt.plot(veval, dragf_epoch0, color = 'red', label = 'epoch 0') # true data
plt.plot(veval, dragf_epochf, color = 'orange', label = 'epoch f') # true data

plt.xlabel('v')
plt.ylabel('dragf')
plt.legend()

plt.savefig('v-dragf.png')
plt.close()
