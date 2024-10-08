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
        self.rn = eval_para['r_dsc']*eval_para['para_1'] # maintain Delta r
        self.rm = eval_para['r_infty']*eval_para['para_1'] # maintain V(infty) ~ 0
        self.ln = eval_para['l_max']
        self.para = eval_para
        self.b_dim = eval_para['batch_dim']

        # l x r
        self.r_dsc = torch.linspace(self.rm/self.rn, self.rm, self.rn)
        self.l_dsc = torch.arange(0,self.ln+1, dtype = torch.int).view(1,-1,1,1)
        
        self.r_dsc_pw = (torch.linspace(1/self.rn, 1, self.rn).view(-1,1).expand(-1,eval_para['pw']+1)**(torch.cat((torch.tensor([-1]), torch.arange(1, eval_para['pw']+1))))).T
    
    def potential_radial_tr(self):
        if self.para['ptl_form'] == "coulomb":
            return -self.para['para_1']/self.r_dsc
        elif self.para['ptl_form'] == "zero":
            return torch.zeros(self.rn)
        else:
            print("Wrong potential type")
    
    def dsc_eigs(self, ptl):
        dsc_lap = (-2*torch.eye(self.rn) + torch.diag(torch.ones(self.rn-1),1) + torch.diag(torch.ones(self.rn-1),-1))*(self.rn/self.rm)**2
        dsc_lap = dsc_lap.view(1, 1, self.rn, self.rn)
        dsc_ptl = torch.diag_embed(ptl).view(self.b_dim, 1, self.rn, self.rn)
        dsc_eff = self.l_dsc*(self.l_dsc+1)*torch.diag(1/self.r_dsc**2).view(1, 1, self.rn,self.rn)
        
        dsc_hmt = (-self.para['para_0']*dsc_lap + dsc_ptl + self.para['para_0']*dsc_eff)

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
                          fin = eval_para['pw'] + 1, act = nn.ReLU())
        self.mlp = nn.Sequential(*modules)


    def forward(self, energy):
        # obtain potential via model
        # via mlp
        coeffs = self.mlp(energy)
        ptl_md = coeffs @ self.r_dsc_pw

        # calculate learned energy
        evl_md = self.dsc_eigs(ptl_md)

        return ptl_md, evl_md


#|%%--%%| <r2Tqi3WSMQ|2mIh88NaIt>

eval_para = {
        # evaluation model
        'r_infty' : 100, # high r_infty maintains V(infty) ~ 0
        'r_dsc' : 1000, # computation time vs accuracy; maintain delta r ~ 0
        'l_max' : 0, # maximum l_max to evaluate radial schrodinger upto
        
        # potential specifics
        'ptl_form' : 'coulomb',
        'para_0' : 1, # hbar^2/2m
        'para_1' : 1, # scales energy; scales horizontally (r_rat) and modelwise (r_dsc)

        # model specifics
        'precision' : 64, # 32 or 64 bit
        'pw' : 5, # consider 1/r and up to r^n
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

#|%%--%%| <2mIh88NaIt|LUnjgpMfmV>

#evl = disc_eigs(ptl, xn)
eval = EvalEig(eval_para)

coeffs_tr = torch.rand(eval_para['batch_dim'], eval_para['pw']+1)*eval_para['para_1']
coeffs_tr[0] = -torch.abs(coeffs_tr[0])
coeffs_tr[:,1:] = torch.tensor([0])
print(coeffs_tr)

ptl_tr = coeffs_tr @ eval.r_dsc_pw
evl_tr = eval.dsc_eigs(ptl_tr)

#|%%--%%| <LUnjgpMfmV|hg9ATHnhv1>

clip_size = evl_tr.shape[2]
for i in range(evl_tr.shape[1]):
    clip_size = min(clip_size, evl_tr[:,i][evl_tr[:,i]<0].shape[0])
clip_size -= 5
print(clip_size)

#print(evl_tr[:,0])
#print(torch.sqrt(1/(evl_tr[:,:clip_size]/evl_tr[:,0])))

#|%%--%%| <hg9ATHnhv1|pdlxDeW0Eu>

model = InvEig(clip_size, eval_para, model_para)
#for submodule in model.modules():
#    submodule.register_forward_hook(nan_hook)
optimiser = torch.optim.Adam(model.parameters(), lr = model_para['lr'])
epochs = model_para['epoch']
pbar = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')
loss_list = [[], []]

for e in range(epochs):
    #ptl_md, evl_md = model(hp['true'])
    #with torch.autograd.detect_anomaly():
    
    ptl_md, evl_md = model(evl_tr[:,:,:clip_size])
    loss_0 = nn.L1Loss()(evl_tr[:,:,:clip_size], evl_md[:, :,:clip_size])

    if e == 0:
        ptl_init = ptl_md

    #loss_1 = torch.sum((ptl_md[1:] - ptl_md[:-1])**2) # smoothness

    loss = loss_0 + model_para['reg1']*loss_1/((e+1)/epochs)**1.5

    loss_list[0].append(loss_0.item())
    loss_list[1].append(loss_1.item())

    #print(loss_0.item(), loss.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    pbar.update()

#|%%--%%| <pdlxDeW0Eu|AjssDzCQlV>

# l11loss
#print(evl_tr[:,:clip_size])
#print(evl_md[:,:clip_size])
print((evl_tr[:,:clip_size]-evl_md[:,:clip_size])/evl_tr[:,:clip_size])

#|%%--%%| <AjssDzCQlV|UGcCcK0clZ>

# ptl true - model
plt.figure()
#plt.plot(eval.r_dsc, eval.potential_radial_tr(), label='true', color = 'red')
#plt.plot(eval.r_dsc, ptl_init.squeeze(0).detach(), label='model_i', color = 'green')
plt.plot(eval.r_dsc, ptl_md[0].squeeze(0).detach(), label='model', color = 'blue')
plt.legend()
plt.show()

#|%%--%%| <UGcCcK0clZ|Vi3de8aRVr>

print(loss_list[0][-1])
plt.figure()
plt.plot(range(epochs), loss_list[0])
plt.show()

#|%%--%%| <Vi3de8aRVr|wwLEsJ77to>

print(loss_list[1][-1])
plt.figure()
plt.plot(range(epochs), loss_list[1])
plt.show()

