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

def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print(outputs)
            print("In ", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i}")

#|%%--%%| <7VzySwJshN|r2Tqi3WSMQ>

class EvalEig(nn.Module):
    def __init__(self, eval_para):
        super().__init__()
        self.rn = eval_para['r_dsc']
        self.rm = eval_para['r_max']
        self.ln = eval_para['l_max']
        self.para = eval_para

        # l x r
        self.r_dsc = torch.linspace(self.rm/self.rn, self.rm, self
.rn)
        self.l_dsc = torch.arange(0,self.ln+1, dtype = torch.int).view(-1,1,1)
    
    def potential_radial_tr(self):
        if self.para['ptl_form'] == "coulomb":
            return -1/self.r_dsc
        elif self.para['ptl_form'] == "zero":
            return torch.zeros(self.rn)
        else:
            print("Wrong potential type")
    
    def dsc_eigs(self, ptl):
        dsc_lap = (-2*torch.eye(self.rn) + torch.diag(torch.ones(self.rn-1),1) + torch.diag(torch.ones(self.rn-1),-1))#*(self.rn/self.rm)**2
        dsc_lap = dsc_lap.view(1, self.rn, self.rn)
        dsc_ptl = torch.diag(ptl).view(1, self.rn, self.rn)
        dsc_eff = self.l_dsc*(self.l_dsc+1)*torch.diag(1/self.r_dsc**2).view(1,self.rn,self.rn)
        
        dsc_hmt = (-dsc_lap/2 + dsc_ptl + dsc_eff)

        #print(dsc_hmt)

        evl, _ = torch.linalg.eigh(dsc_hmt)
        # bound state condition
        # evl[evl>0] = torch.tensor(0)

        return evl

class InvEig(EvalEig):
    def __init__(self, clip_size, eval_para, model_para):
        super().__init__(eval_para)
        # model
        modules = seq_mlp(init = clip_size, mlp = model_para['mlp'],
                          fin = self.rn, act = nn.ReLU())

        self.mlp = nn.Sequential(*modules)
 
    def forward(self, energy, ptl_func = None):
        # obtain potential via model
        # via mlp
        ptl_md = self.mlp(energy)
        
        # calculate learned energy
        evl_md = self.dsc_eigs(ptl_md)

        return ptl_md, evl_md


#|%%--%%| <r2Tqi3WSMQ|2mIh88NaIt>

eval_para = {
        # evaluation model
        'r_max' : 10000, # horizontal scaling of model
        'r_dsc' : 1000, # computation time vs accuracy
        'l_max' : 1, # maximum l_max to evaluate radial schrodinger upto
        
        # potential specifics
        'ptl_form' : 'coulomb',
        'para_1' : 1, # potential multiplier (sommerfield parameter); Ze^2m/hbar^2 in desired units; vertical scaling of model

        # model specifics
        'precision' : 64, # 32 or 64 bit
        }

model_para = {
        # model
        'mlp' : [1000, 500, 1000],

        # training
        'epoch' : 1000,
        'lr' : 1e-2,

        # loss regularisation
        'reg1' : 1, # smoothness
        }

#|%%--%%| <2mIh88NaIt|LUnjgpMfmV>

#evl = disc_eigs(ptl, xn)
eval = EvalEig(eval_para)
evl_tr = eval.dsc_eigs(eval.potential_radial_tr())

clip_size = evl_tr.shape[1]
for i in range(evl_tr.shape[0]):
    clip_size = min(clip_size, evl_tr[i][evl_tr[i]<0].shape[0])

#|%%--%%| <LUnjgpMfmV|GObqqyxWon>

#print(evl_tr)
print(clip_size)
print(evl_tr[:,:clip_size])

#|%%--%%| <GObqqyxWon|pdlxDeW0Eu>

model = InvEig(clip_size, eval_para, model_para)
#for submodule in model.modules():
#    submodule.register_forward_hook(nan_hook)

optimiser = torch.optim.Adam(model.parameters(), lr = model_para['lr'])

epochs = model_para['epoch']

pbar = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')

for e in range(epochs):
    #ptl_md, evl_md = model(hp['true'])
    #with torch.autograd.detect_anomaly():
    
    loss_0 = torch.tensor([0.])
    for i in range(evl_tr.shape[0]):
        ptl_md, evl_md = model(evl_tr[i,:clip_size])
        loss_0 += nn.L1Loss()(evl_tr[i,:clip_size], evl_md[i,:clip_size])

    loss_1 = torch.sum((ptl_md[1:] - ptl_md[:-1])**2)

    loss = loss_0 + model_para['reg1']*loss_1/((e+1)/epochs)**1.2

    print(loss_0.item(), loss.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    pbar.update()

#|%%--%%| <pdlxDeW0Eu|AjssDzCQlV>

# l11loss
print(evl_tr)
print(evl_md)

#|%%--%%| <AjssDzCQlV|UGcCcK0clZ>

# ptl true - model
plt.figure()
plt.plot(eval.r_dsc, eval.potential_radial_tr(), label='true', color = 'red')
#plt.plot(eval.x_dsc, ptl_init.squeeze(0).detach(), label='model_i', color = 'green')
plt.plot(eval.r_dsc, ptl_md.squeeze(0).detach(), label='model', color = 'blue')
#plt.plot(eval.x_spl, torch.cat((para_0, torch.tensor([0.]))).detach(), color = 'red', marker = 'o')
plt.legend()
plt.show()

#|%%--%%| <UGcCcK0clZ|Vi3de8aRVr>

print(nn.L1Loss()(eval.ptl_tr(hp['true']), ptl_md))

#|%%--%%| <Vi3de8aRVr|tgFRr8WZLe>


