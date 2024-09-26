import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

# implement some way to track units and scaling constants
# check if python lsp is working?

def seq_mlp(init, mlp, fin, act):
    modules = [nn.Linear(init, mlp[0]), act]
    for i in range(len(mlp) - 1):
        modules.append(nn.Linear(mlp[i], mlp[i+1]))
        modules.append(act)

    modules.append(nn.Linear(mlp[-1], fin)) #self.spl for spline

    return modules

# eigenvalue encoding

def boundary_cond(ptl, cnd):
    if cnd == 'inf':
        ptl[0] = hp['b_inf']
        ptl[-1] = hp['b_inf']
    
    if cnd == 'zero':
        ptl[0] = 0
        ptl[-1] = 0

    return ptl

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
    def __init__(self, hp):
        super().__init__()
        self.xn = hp['x_dsc']
        self.xm = hp['x_max']
        self.spl = hp['spline']
        self.hp = hp

        self.xn_tr = int(self.xn*hp['cutoff'])
        self.x_dsc = torch.linspace(-self.xm, self.xm, self.xn)
        self.x_spl = torch.linspace(0, self.xm, self.spl)

        self.loss_l = [[], [], [], [], [], []] # loss_0 : total loss
    
    def graph_evc(self, evl, evc):
        colors = plt.cm.viridis(np.linspace(0, 1, self.xn_tr))
        plt.figure()
        for i in range(int(self.xn_tr)):
            plt.plot(self.x_dsc, evc[i], label = 'ev : ' + str(round(evl[i].item(), 2)), color = colors[i])
        plt.legend()
        plt.show()
    
    def evl_enc(self, evl):
        diffs = evl[1:] - evl[:-1]
        diffs = (diffs/diffs[0]) ** 4 # amplify
        diffs[0] = evl[0]
        return diffs

    def discrete_lap(self): 
        xd = 2 * self.xm / (self.xn - 1)
        cns = 1/xd**2 * (-1/2)
        lap = -2*torch.eye(self.xn) + torch.diag(torch.ones(self.xn-1),1) + torch.diag(torch.ones(self.xn-1),-1) # circle graph
        #lap[:0][:0], lap[-1:][-1:] = 1, 1 # line graph
        return lap * cns

    def ptl_tr(self, ptl_func):
        if ptl_func == "sho":
            ptl = (self.x_dsc**2/2 - self.xm**2/2)*self.hp['scale']
        elif ptl_func == "zero":
            ptl = torch.zeros(self.xn)
        elif ptl_func == "wedge":
            ptl = (torch.abs(self.x_dsc) - self.xm)*self.hp['scale']
        elif ptl_func == "sombrero":
            ptl = ((-3*self.x_dsc**2+self.x_dsc**4)-(-3*self.xm**2+self.xm**4))*self.hp['scale']
        elif ptl_func == "coulomb":
            ptl = -1/self.x_dsc**2*self.hp['scale']

        return ptl
    
    def disc_prob(self, evc):
        tr_mtrx = torch.linalg.inv(evc)
        dir_dlt = torch.zeros(self.xn)
        dir_dlt[self.xn//2] = 1
        probs = torch.matmul(dir_dlt, tr_mtrx)**2
        return probs

    def disc_eigs(self, ptl):
        hml = (self.discrete_lap() + torch.diag(ptl))
        evl, evc = torch.linalg.eigh(hml)
        #print(torch.min(torch.abs(evl[1:]-evl[:-1])))
        evl = evl[:self.xn_tr] # add condition to prevent positive energies
        #evl = evl_enc(evl)
        prob = eval.disc_prob(evc.T)[:self.xn_tr]

        return evl, prob

class InvEig(EvalEig):
    def __init__(self, energy, hp):
        super().__init__(hp)
        # take energies as input of model; use info to define form of model?
        self.evl = energy
        #self.prob = torch.log(prob)

        # model
        modules = seq_mlp(init = self.evl.shape[0], mlp = hp['mlp'],
                          fin = self.xn, act = nn.ReLU())
        #modules = seq_mlp(init = self.evl.shape[0]*2, mlp = hp['mlp'],
                          #fin = self.spl - 1, act = nn.ReLU()

        self.mlp = nn.Sequential(*modules)
        #self.mlp = nn.Parameter(torch.zeros([self.evl.shape[0],self.xn]))

        #self.paras = nn.Parameter(torch.rand(self.spl).unsqueeze(0).T)

    def spline(self, para):
        t = torch.linspace(-self.xm, self.xm, 2*self.spl - 1)
        coeffs = natural_cubic_spline_coeffs(t, para)
        spline = NaturalCubicSpline(coeffs)
        return spline.evaluate(torch.linspace(-self.xm, self.xm, self.xn))
    
    def loss_calc(self, ptl_md, evl_md, e):
        loss = torch.tensor([0.])
        # l1loss
        loss1 = nn.L1Loss()(self.evl, evl_md)/torch.abs(self.evl[0])
                           # consider loss that increases weight for ground state
        # variation; 'smoothness'
        loss2 = torch.std(ptl_md)
        # potential ground energy condition
        loss3 = torch.abs(torch.max(self.evl[0] - ptl_md))/torch.abs(self.evl[0])

        #loss4 = nn.L1Loss()(self.prob, torch.log(prob_md))

        loss4 = torch.sum((ptl_md[1:]-ptl_md[:-1])**2)/self.evl[0]**2

        loss5 = (ptl_md[0]**2 + ptl_md[-1]**2)/self.evl[0]**2

        loss6 = ptl_md[self.xn//2]/torch.abs(self.evl[0])

        loss = self.hp['reg_1'] * loss1 + \
               self.hp['reg_3'] * loss3 + \
               self.hp['reg_4'] * loss4 + \
               self.hp['reg_5'] * loss5 + \
               self.hp['reg_6'] * loss6
               #self.hp['reg_4'] * loss4 #/ ((e+1)/self.hp['epoch'])**1.5 

        self.loss_l[0].append(loss.item())
        self.loss_l[1].append(loss1.item())
        self.loss_l[2].append(loss1.item())
        self.loss_l[3].append(loss3.item())
        self.loss_l[4].append(loss4.item())
        self.loss_l[5].append(loss5.item())

        return loss

    def loss_plot(self, index, ignore):
        print(self.loss_l[index][-1])
        plt.figure() #l1loss
        plt.plot(range(hp['epoch']-ignore), self.loss_l[index][ignore:])
        plt.show()

    def forward(self, ptl_func = None):
        # obtain potential via model
        # via mlp
        ptl_md = self.mlp(self.evl)
        #ptl_raw = self.mlp(self.evl).unsqueeze(0)
        #ptl = self.smooth(ptl_raw).squeeze(0)
        
        # via spline; symmetric
        if ptl_func == "sho":
            para_0 = nn.Parameter((torch.linspace(0,self.xm,self.spl)[:-1]**2/2-self.xm**2/2)*self.hp['scale']) + self.mlp(self.evl)
        elif ptl_func == "zer":
            para_0 = nn.Parameter(torch.zeros(self.spl-1))
        #else:
            # usually negative monotonely increasing
            #para_ran = -torch.abs(self.mlp(torch.cat((self.evl, self.prob))))
            #ground = para_ran[0]
            #pos = (para_ran[1:] - torch.mean(para_ran[1:]))/torch.std(para_ran[1:])*ground/self.spl
            #para_0 = torch.cat((torch.tensor([ground]), ground + torch.cumsum(torch.exp(pos), dim = 0)))  #: exp
            #para_0 = torch.cat((torch.tensor([ground]), ground + torch.cumsum(torch.abs(pos), dim = 0)))
            # para_0 = self.mlp(torch.cat((self.evl, self.prob)))

        #para_1 = torch.cat((para_0, torch.tensor([0.])))
        #para_2 = para_1.flip(0)
        #spline_para = torch.cat((para_2, para_1[1:])).unsqueeze(0).T
        #ptl_md = self.spline(spline_para).squeeze(1)

        # fourier
        #coeffs = torch.zeros(self.xn)
        #coeffs[:self.spl-1] = torch.complex(self.mlp(self.evl),
        #                       torch.zeros(self.spl-1))
        #ptl_md = torch.fft.ifft(coeffs).real

        # calculate learned energy
        evl_md, prob_md = self.disc_eigs(ptl_md)

        #return ptl_md, evl_md, prob_md, para_0
        return ptl_md, evl_md


#|%%--%%| <r2Tqi3WSMQ|2mIh88NaIt>

hp = {
        # physical model
        'x_max' : 4, # roughly : exp -xm**2 ~ 0
        'x_dsc' : 200, # computation time for diagonalisation; consider sparse matrix
        'cutoff' : 0.5, # ratio of evls to consider

        'true' : 'sho',
        'scale' : 0.01, # scale of eigenvalues etc.

        # model specs
        'mlp' : [1000, 500, 1000],
        'spline' : 10,

        # training
        'epoch' : 1000,
        'lr' : 1e-2,
        'reg_1' : 1, # fix
        'reg_2' : 0, # stddev
        'reg_3' : 1e-2, # ge-ptl
        'reg_4' : 1e-2, # smoothness
        'reg_5' : 1, # bc
        'reg_6' : 0, # potential at zero
        }

#xn = int(2*xm/xd + 1)
eval = EvalEig(hp)

#|%%--%%| <2mIh88NaIt|LUnjgpMfmV>

#evl = disc_eigs(ptl, xn)
evl_tr, prob_tr = eval.disc_eigs(eval.ptl_tr(hp['true']))
print(evl_tr)

#ns = torch.linspace(1, 100, 100)
#evl_tr = -1/ns**2
#print(evl_tr)

#|%%--%%| <LUnjgpMfmV|GObqqyxWon>

#eval.graph_evc(evl_tr, evc)

#|%%--%%| <GObqqyxWon|pdlxDeW0Eu>

model = InvEig(evl_tr, hp)
#for submodule in model.modules():
#    submodule.register_forward_hook(nan_hook)

optimiser = torch.optim.Adam(model.parameters(), lr = hp['lr'])

epochs = hp['epoch']

pbar = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')

for e in range(epochs):
    #ptl_md, evl_md = model(hp['true'])
    with torch.autograd.detect_anomaly():
        #ptl_md, evl_md, prob_md, para_0 = model()
        ptl_md, evl_md = model()
        if e == 0:
            ptl_init = ptl_md
            evl_init = evl_md
        # constraints via loss : l1, stdev, ge-ptl
        # constraints via model : smooth, symmetry, bc
        loss = model.loss_calc(ptl_md, evl_md, e)
        
        #print(torch.sum(prob_md))

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    pbar.update()

#|%%--%%| <pdlxDeW0Eu|ubKkTJobNY>cubic splinetor

# total loss
model.loss_plot(index = 0, ignore = 4)

#|%%--%%| <ubKkTJobNY|AjssDzCQlV>

# l11loss
model.loss_plot(index = 1, ignore = 0)

#|%%--%%| <AjssDzCQlV|d8uQM0ydA8>

model.loss_plot(index = 4, ignore = 0)

#|%%--%%| <d8uQM0ydA8|RICrt6vY3O>

model.loss_plot(index = 5, ignore = 0)

#|%%--%%| <RICrt6vY3O|wiXrFRTr6y>

# energy true - model
plt.figure()
plt.plot(range(eval.xn_tr), evl_tr, label='true', color = 'red')
#plt.plot(range(eval.xn_tr), evl_init.detach(), label='model_i', color = 'green')
plt.plot(range(eval.xn_tr), evl_md.detach(), label='model', color = 'blue')
plt.legend()
plt.show()

#|%%--%%| <wiXrFRTr6y|UGcCcK0clZ>

# ptl true - model
plt.figure()
plt.plot(eval.x_dsc, eval.ptl_tr(hp['true']), label='true', color = 'red')
#plt.plot(eval.x_dsc, ptl_init.squeeze(0).detach(), label='model_i', color = 'green')
plt.plot(eval.x_dsc, ptl_md.squeeze(0).detach(), label='model', color = 'blue')
#plt.plot(eval.x_spl, torch.cat((para_0, torch.tensor([0.]))).detach(), color = 'red', marker = 'o')
plt.legend()
plt.show()

#|%%--%%| <UGcCcK0clZ|Vi3de8aRVr>

print(nn.L1Loss()(eval.ptl_tr(hp['true']), ptl_md))

#|%%--%%| <Vi3de8aRVr|tgFRr8WZLe>

