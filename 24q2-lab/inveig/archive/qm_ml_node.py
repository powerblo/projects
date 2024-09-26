import math
import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
from tqdm.auto import tqdm

# implement some way to track units and scaling constants
# check if python lsp is working?

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
        self.para = eval_para
        self.rm = self.para['r_max']
        self.rn = self.para['r_dsc']
        
        self.r_dsc = torch.linspace(self.rm/self.rn, self.rm, self.rn).view(-1,1,1)
        self.l_dsc = torch.arange(0,self.para['l_max']+1, dtype = torch.int).view(1,1,-1)

        if eval_para['precision'] == 64:
            torch.set_default_dtype(torch.float64)

        self.loss_l = [[], [], [], [], [], []] # loss_0 : total loss

    def plot(self, funclist, indices = None):
        plt.figure()
        for i in range(len(funclist)):
            if indices == None:
                plt.plot(range(funclist[i].shape[0]), funclist[i]/torch.max(torch.abs(funclist[i])), label=i)
            else:
                plt.plot(torch.arange(indices[i][0], indices[i][1]), funclist[i][indices[i][0]:indices[i][1]]/torch.max(torch.abs(funclist[i][indices[i][0]:indices[i][1]])), label=i)
        plt.legend()
        plt.show()
    
    def potential_radial_tr(self):
        if self.para['ptl_form'] == "coulomb":
            return -1/self.r_dsc
        else:
            return torch.zeros(self.rn)

    def diffeq_factor(self, energy, ptl):
        v_val = ptl.view(-1,1,1)
        e_val = energy.view(1,energy.shape[0],energy.shape[1])

        # rn x en x ln
        factor = self.l_dsc*(self.l_dsc+1)/self.r_dsc**2 + self.para['para_1']*(-e_val + v_val)
        
        return e_val, factor

    def disc_solve(self, sol, factor, limit):
        # sovles sol^(2) = factor * sol
        rd = self.rm/self.rn
        if self.para['step'] == 4:
            transf = torch.zeros(limit - 5,5,5, factor.shape[1], factor.shape[2])
            sols = torch.zeros(limit - 5, 5, factor.shape[1], factor.shape[2])
            sols[0] = sol
            for i in range(1, limit - 5):
                transf[i][0,1,:,:] = 1
                transf[i][1,2,:,:] = 1
                transf[i][2,3,:,:] = 1
                transf[i][3,4,:,:] = 1
                transf[i][4,1] = -1 + 3/40*rd**2*factor[i]/(1 - 3/40*rd**2*factor[i+4])
                transf[i][4,2] = 2 + 13/15*rd**2*factor[i+1]/(1 - 3/40*rd**2*factor[i+4])
                transf[i][4,3] = -2 + 7/60*rd**2*factor[i+2]/(1 - 3/40*rd**2*factor[i+4])
                transf[i][4,4] = 2 + 13/15*rd**2*factor[i+3]/(1 - 3/40*rd**2*factor[i+4])

                sols[i] = torch.einsum('ijab,jab->iab', transf[i], sols[i-1])
        return sols
    
    def derivative(self, func, at):
        rd = self.rm/self.rn
        if self.para['step'] == 4:
            return (25*func[at-1] - 48*func[at-2] + 36*func[at-3] - 16*func[at-4] + 3*func[at-5])/(12*rd)

    def integrate(self, func):
        rd = self.rm/self.rn
        integrand = torch.zeros(func.shape[0]//4-1, func.shape[1], func.shape[2])
        for i in range(0,integrand.shape[0]):
            integrand[i, :,:] = (7*(func[4*i]**2+func[4*i+4]**2)+
                         32*(func[4*i+1]**2+func[4*i+3]**2)+12*func[4*i+2])*(2*rd/45)
        
        integral = torch.sum(integrand, dim = 0)

        return integral
     
    def dual_propagate(self, e_val, factor):
        r_zero_init = torch.linspace(self.rm/self.rn, self.rm/self.rn*5, 5).view(-1,1,1)
        ptl_m1 = -self.para['para_1'] # coulomb case
        # r^(l+1) condition + first order expansion
        u_zero_init_unsize = ((r_zero_init ** (self.l_dsc+1)) + (r_zero_init ** (self.l_dsc+2) / (2*(self.l_dsc+1)) * ptl_m1))
        u_zero_init = u_zero_init_unsize.expand(-1,factor.shape[1],-1)
        
        r_infty_init = torch.linspace(self.rm - 4*self.rm/self.rn, self.rm, 5).view(-1,1,1)
        factorial_term = torch.tensor([math.prod([2 * l + 1 for l in range(l_m + 1)]) for l_m in self.l_dsc.squeeze()])
        base_term = (r_infty_init * torch.sqrt(torch.abs(e_val))) ** self.l_dsc / factorial_term
        series_term_0 = torch.ones_like(base_term)
        series_term_1 = ((r_infty_init * torch.sqrt(torch.abs(e_val)))**2/2)/(2*self.l_dsc+3)
        series_term_2 = ((r_infty_init * torch.sqrt(torch.abs(e_val)))**2/2)**2/(2*(2*self.l_dsc+3)*(2*self.l_dsc+5))
        bessel = base_term * (series_term_0 + series_term_1 + series_term_2)
        u_infty_init = r_infty_init * bessel
        
        matching_radius = int(self.rn*self.para['matching_radius'])
        u_zero_list, u_infty_list = (self.disc_solve(u_zero_init, factor, matching_radius),
                           self.disc_solve(u_infty_init, factor.flip(0), self.rn - matching_radius).flip(0))
        del u_zero_init, u_infty_init
        u_zero = torch.concat((u_zero_list[0,:,:,:], u_zero_list[:,4,:,:]), dim = 0)
        u_infty = torch.concat((u_infty_list[0,:,:,:], u_infty_list[:,4,:,:]), dim = 0)
        
        return u_zero, u_infty

    def energy_step(self, u_zero, u_infty):
        # dynamic matching radii as function of E, l
        # fix matching radii
        matching_radius = int(self.rn*self.para['matching_radius'])
        lfunc_in = self.derivative(u_zero, matching_radius)/u_zero[-1]
        lfunc_out = self.derivative(u_infty.flip(0), self.rn - matching_radius).flip(0)/u_infty[0]

        integ_in = self.integrate(u_zero**2)
        integ_out = self.integrate(u_infty**2)

        delta_energy = -(lfunc_out - lfunc_in) / (integ_in/u_zero[-1]**2 + 
                                                  integ_out/u_infty[0]**2)
        return delta_energy
    
    def disc_eig(self, init_energy):
        energy = init_energy.view(-1,1).expand(-1,eval_para['l_max']+1).clone()
        energy_step = torch.ones_like(energy)
        #while(torch.max(torch.abs(energy_step)) > self.para['cutoff']):
        print("----")
        e_val, factor = self.diffeq_factor(energy, self.potential_radial_tr())
        print('e_val : ', e_val)
        print('factor : ', factor)
        u_zero, u_infty = self.dual_propagate(e_val, factor)
        print('u_zero : ', u_zero)
        print('u_infty : ', u_infty)

        energy_step = self.energy_step(u_zero, u_infty)
        print('energy_step : ', energy_step)
        energy = energy + energy_step

        return energy, u_zero, u_infty

class InvEig(EvalEig):
    def __init__(self, energy, eval_para):
        super().__init__(eval_para)
        self.energy = energy

        self.ptl = nn.Parameter(torch.rand(self.rn))

    def forward(self):
        energy_sample = self.energy # consider random samping a la monte carlo
        e_val, factor = self.diffeq_factor(energy_sample, self.ptl)
        u_zero, u_infty = self.dual_propagate(e_val, factor)

        loss = torch.mean(torch.abs(self.energy_step(u_zero, u_infty)))

        return loss

#|%%--%%| <r2Tqi3WSMQ|2mIh88NaIt>

eval_para = {
        # evaluation model
        'r_max' : 100, # horizontal scaling of model
        'r_dsc' : 1000, # computation time vs accuracy
        'l_max' : 2, # maximum l_max to evaluate radial schrodinger upto
        'matching_radius' : 0.1, # ratio of matching radius to total

        # potential specifics
        'ptl_form' : 'coulomb',
        'para_0' : 1, # laplacian multiplier; vertical scaling of model
        'para_1' : 1, # potential multiplier (sommerfield parameter); Ze^2m/hbar^2 in desired units

        # model specifics
        'precision' : 64, # 32 or 64 bit
        'step' : 4, # 2 or 4
        'cutoff' : 1e-6,
        }

model_para = {
        # training
        'epoch' : 1000,
        'lr' : 1e-2,
        }

#|%%--%%| <2mIh88NaIt|LUnjgpMfmV>

eval = EvalEig(eval_para)

#evl = disc_eigs(ptl, xn)
energy, u_zero, u_infty = eval.disc_eig(torch.linspace(-0.06, -0.008, 30))

#|%%--%%| <LUnjgpMfmV|GObqqyxWon>

print(u_zero)

#|%%--%%| <GObqqyxWon|LDeARqgitp>

print(1/(math.sqrt(0.0605/0.0263)-1))
print(1/(math.sqrt(0.0263/0.0146)-1))
print(1/(math.sqrt(0.0146/0.0082)-1))

#|%%--%%| <LDeARqgitp|pdlxDeW0Eu>

model = InvEig(energy, eval_para)
#for submodule in model.modules():
#    submodule.register_forward_hook(nan_hook)

optimiser = torch.optim.Adam(model.parameters(), lr = model_para['lr'])

epochs = model_para['epoch']

pbar = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')

for e in range(epochs):
    #ptl_md, evl_md = model(hp['true'])
    with torch.autograd.detect_anomaly():
        #ptl_md, evl_md, prob_md, para_0 = model()
        loss = model()
        # constraints via loss : l1, stdev, ge-ptl
        # constraints via model : smooth, symmetry, bc
        #print(torch.sum(prob_md))

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    pbar.update()

#|%%--%%| <pdlxDeW0Eu|AjssDzCQlV>

# l11loss
model.loss_plot(index = 1, ignore = 5)

#|%%--%%| <AjssDzCQlV|wiXrFRTr6y>

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
plt.plot(eval.r_dsc, eval.potential_radial_tr(), label='true', color = 'red')
#plt.plot(eval.x_dsc, ptl_init.squeeze(0).detach(), label='model_i', color = 'green')
plt.plot(eval.r_dsc, ptl_md.squeeze(0).detach(), label='model', color = 'blue')
#plt.plot(eval.x_spl, torch.cat((para_0, torch.tensor([0.]))).detach(), color = 'red', marker = 'o')
plt.legend()
plt.show()

#|%%--%%| <UGcCcK0clZ|Vi3de8aRVr>

print(nn.L1Loss()(eval.ptl_tr(hp['true']), ptl_md))

#|%%--%%| <Vi3de8aRVr|tgFRr8WZLe>

