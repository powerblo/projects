import math
import torch, torch.nn as nn
import torchode as to
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
    def __init__(self, eval_para, e_dsc):
        super().__init__()
        self.para = eval_para
        self.rm = self.para['r_max']
        self.rn = self.para['r_dsc']

        self.e_dsc = e_dsc
        
        self.r_dsc = torch.linspace(self.rm/self.rn, self.rm, self.rn).view(-1,1,1)
        self.l_dsc = torch.arange(0,self.para['l_max']+1, dtype = torch.int).view(1,1,-1)

        if eval_para['precision'] == 64:
            torch.set_default_dtype(torch.float64)

        self.loss_l = [[], [], [], [], [], []] # loss_0 : total loss

    def plot(self, funclist, indices = None):
        plt.figure()
        for i in range(len(funclist)):
            if indices == None:
                #plt.plot(range(funclist[i].shape[0]), funclist[i]/torch.max(torch.abs(funclist[i])), label=i)
                plt.plot(range(funclist[i].shape[0]), funclist[i], label=i)
            else:
                plt.plot(torch.arange(indices[i][0], indices[i][1]), funclist[i][indices[i][0]:indices[i][1]]/torch.max(torch.abs(funclist[i][indices[i][0]:indices[i][1]])), label=i)
        plt.legend()
        plt.show()
    
    def potential_radial_tr(self):
        if self.para['ptl_form'] == "coulomb":
            return -1/self.r_dsc
        elif self.para['ptl_form'] == "zero":
            return torch.zeros(self.rn)
        else:
            return torch.zeros(self.rn)

    def diffeq_factor(self, energy, ptl):
        v_val = ptl.view(-1,1,1)
        e_val = energy.view(1,energy.shape[0],energy.shape[1])

        # rn x en x ln
        factor = self.l_dsc*(self.l_dsc+1)/self.r_dsc**2 + self.para['para_1']*(-e_val + v_val)
        
        return e_val, factor

    def derivative(self, func, at):
        rd = self.rm/self.rn
        if self.para['step'] == 4:
            return (25*func[at] - 48*func[at-1] + 36*func[at-2] - 16*func[at-3] + 3*func[at-4])/(12*rd)

    def integrate(self, func, i, f):
        rd = self.rm/self.rn
        integral = torch.zeros(func.shape[1], func.shape[2])
        for i in range((f - i)//4 - 1):
            integral += (7*(func[4*i]**2+func[4*i+4]**2)+
                         32*(func[4*i+1]**2+func[4*i+3]**2)+12*func[4*i+2])*(2*rd/45)
        return integral
     
    def dual_propagate(self, e_val, factor): 
        #u_init = self.r_dsc.view(-1,1,1).expand(self.rn, factor.shape[1], self.para['l_max'])
        ptl_m1 = -self.para['para_1'] # coulomb case
        # r^(l+1) condition + first order expansion
        u_zero_init_unsize = ((self.r_dsc ** (self.l_dsc+1)) + (self.r_dsc ** (self.l_dsc+2) / (2*(self.l_dsc+1)) * ptl_m1))
        u_zero_init = u_zero_init_unsize.expand(-1,factor.shape[1],-1)

        if self.l_dsc.shape[2] != 1:
            factorial_term = torch.tensor([math.prod([2 * l + 1 for l in range(l_m + 1)]) for l_m in self.l_dsc.squeeze()])
        else:
            factorial_term = 1
        base_term = (self.r_dsc * torch.sqrt(torch.abs(e_val))) ** self.l_dsc / factorial_term
        series_term_0 = torch.ones_like(base_term)
        series_term_1 = ((self.r_dsc * torch.sqrt(torch.abs(e_val)))**2/2)/(2*self.l_dsc+3)
        series_term_2 = ((self.r_dsc * torch.sqrt(torch.abs(e_val)))**2/2)**2/(2*(2*self.l_dsc+3)*(2*self.l_dsc+5))
        bessel = base_term * (series_term_0 + series_term_1 + series_term_2)
        u_infty_init = self.r_dsc * bessel
        
        # dynamic matching radii as function of E, r
        # fix matching radii
        matching_radius = int(self.rn*self.para['matching_radius'])
        u_zero, u_infty = (self.disc_solve(u_zero_init, factor),
                           self.disc_solve(u_infty_init, factor.flip(0)).flip(0))
        del u_zero_init, u_infty_init
        
        return u_zero, u_infty

    def energy_step(self, u_zero, u_infty):
        # dynamic matching radii as function of E, l
        # fix matching radii
        matching_radius = int(self.rn*self.para['matching_radius'])
        lfunc_in = self.derivative(u_zero, matching_radius)/u_zero[matching_radius]
        lfunc_out = self.derivative(u_infty, matching_radius)/u_infty[matching_radius]
        
        integ_in = self.integrate(u_zero, 0, matching_radius)
        integ_out = self.integrate(u_infty, matching_radius, self.rn)

        delta_energy = -(lfunc_out - lfunc_in) / (integ_in/u_zero[matching_radius]**2 + 
                                                  integ_out/u_infty[matching_radius]**2)
        return delta_energy

    def step_func(self, ptl):
        energy = self.e_dsc.view(-1,1).expand(-1,eval_para['l_max']+1).clone()
        e_val, factor = self.diffeq_factor(energy, ptl)
        u_zero, u_infty = self.dual_propagate(e_val, factor)

        energy_step_func = self.energy_step(u_zero, u_infty)
        energy_step_smooth = nn.functional.pad(energy_step_func.T, (1,1), mode = 'replicate').T.unfold(dimension=0,size=3,step=1).mean(dim=2)

        guesses_ind = torch.tensor([])

        for i in range(energy_step_smooth.shape[0]-1):
            if (torch.any((energy_step_smooth[i+1]*energy_step_smooth[i] < 0) & (energy_step_smooth[i+1] < energy_step_smooth[i]))):
                guesses_ind = torch.cat((guesses_ind, torch.tensor([i])), dim = 0)

        guesses = self.e_dsc[0]+guesses_ind*(self.e_dsc[-1]-self.e_dsc[0])/(self.e_dsc.shape[0]-1)

        return energy_step_smooth, guesses

    def disc_eig(self, step_func):
        energy = self.e_dsc.view(-1,1).expand(step_func.shape[0],step_func.shape[1]).clone()

        energy_step = step_func.view(1,step_func.shape[0],step_func.shape[1]).clone()
        
        for _ in range(10):
            energy_clamp = torch.clamp((energy - self.e_dsc[0])*(self.e_dsc.shape[0]-1)/(self.e_dsc[-1] - self.e_dsc[0]), min = 0, max = self.e_dsc.shape[0]-2)
            indices = torch.floor(energy_clamp).long()
            temp_step = torch.zeros_like(step_func)

            for i in range(energy.shape[1]): # over l
                temp_step[:,i] = step_func[:,i][indices[:,i]] * (energy_clamp[:,i] - indices[:,i]) + step_func[:,i][indices[:,i]+1] * (indices[:,i] + 1 - energy_clamp[:,i])

            energy_step = torch.cat((energy_step, temp_step.view(1,step_func.shape[0],step_func.shape[1])), dim = 0)
            energy = energy + energy_step[-1]

        return energy

    def disc_energy_evolution(self, ptl, energy):
        energy_tns = energy.view(-1,1).expand(energy.shape[0],ptl.shape[1]).clone()
        
        e_val, factor = self.diffeq_factor(energy_tns, ptl)
        u_zero, u_infty = self.dual_propagate(e_val, factor)
        energy_step = self.energy_step(u_zero, u_infty)

        return energy_step


class InvEig(EvalEig):
    def __init__(self, eval_para, e_dsc):
        super().__init__(eval_para, e_dsc)
        
        #self.ptl = nn.Parameter(self.potential_radial_tr() + torch.rand(self.rn, 1, 1)/10)
        self.paras = nn.Parameter(torch.rand(self.rn+1, 1, 1))

    def forward(self, energy):
        ptl_func = (-torch.sum(torch.abs(self.paras[1:]), dim = 0) + torch.cumsum(torch.abs(self.paras[1:]), dim = 0))*self.paras[0]/np.sqrt(self.rn)
        #ptl_func = self.ptl
        energy_step = self.disc_energy_evolution(ptl_func, energy)

        return energy_step, ptl_func


#|%%--%%| <r2Tqi3WSMQ|2mIh88NaIt>

eval_para = {
        # evaluation model
        # heuristics : true eigenvalues are attractors; outside of this region extremely low

        'r_max' : 400, # horizontal scaling of model
        # heuristics : low r_max -> low energies
        # ~ 400 : n = 3 ~ 9
         
        'r_dsc' : 2000, # computation time vs accuracy
        # heuristics : roughly degree of r_dsc ~ degree of accuracy; 5000 roughly good

        'l_max' : 0, # maximum l_max to evaluate radial schrodinger upto
        'matching_radius' : 0.1, # ratio of matching radius to total

        # potential specifics
        'ptl_form' : 'coulomb',
        'para_0' : 1, # laplacian multiplier; vertical scaling of model
        'para_1' : 1, # potential multiplier (sommerfield parameter); Ze^2m/hbar^2 in desired units

        # model specifics
        'precision' : 64, # 32 or 64 bit
        'step' : 4, # 2 or 4
        }

model_para = {
        # training
        'epoch' : 10,
        'lr' : 1,
        }

#|%%--%%| <2mIh88NaIt|LUnjgpMfmV>

interval = torch.linspace(-0.03,-0.005,eval_para['r_dsc'])
eval = EvalEig(eval_para, interval)

#evl = disc_eigs(ptl, xn)
energy_step, guesses = eval.step_func(eval.potential_radial_tr())
 
#|%%--%%| <LUnjgpMfmV|j4JPCTVboQ>

print(guesses)

eval.plot([energy_step[:,0]])

#|%%--%%| <j4JPCTVboQ|LDeARqgitp>

guesses_pad = torch.cat((guesses, guesses - (interval[-1]-interval[0])/eval.rn, guesses + (interval[-1]-interval[0])/eval.rn)).view(3,guesses.shape[0]).T.contiguous().view(-1)
energy_step_tr = eval.disc_energy_evolution(eval.potential_radial_tr(), guesses_pad)

#|%%--%%| <LDeARqgitp|pdlxDeW0Eu>

model = InvEig(eval_para, interval)
#for submodule in model.modules():
#    submodule.register_forward_hook(nan_hook)

optimiser = torch.optim.Adam(model.parameters(), lr = model_para['lr'])

epochs = model_para['epoch']

pbar = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')

for e in range(epochs):
    #ptl_md, evl_md = model(hp['true'])
    #with torch.autograd.set_detect_anomaly(True):
        #ptl_md, evl_md, prob_md, para_0 = model()
    energy_step_md, ptl_func = model(guesses_pad)
    if e == 0:
        ptl_init = ptl_func.clone()
    loss = nn.L1Loss()(energy_step_md, energy_step_tr)
    print(loss.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    pbar.update()

#|%%--%%| <pdlxDeW0Eu|AjssDzCQlV>

model.plot([ptl_func[:,0,0].detach()])

#|%%--%%| <AjssDzCQlV|wiXrFRTr6y>

# energy true - model
plt.figure()
plt.plot(range(eval.rn), ptl_init[:,0,0].detach(), label='true', color = 'red')
#plt.plot(range(eval.xn_tr), evl_init.detach(), label='model_i', color = 'green')
plt.plot(range(eval.rn), ptl_func[:,0,0].detach(), label='model', color = 'blue')
plt.legend()
plt.show()

#|%%--%%| <wiXrFRTr6y|UGcCcK0clZ>

# ptl true - model
plt.figure()
plt.plot(eval.r_dsc.squeeze(1,2), eval.potential_radial_tr().squeeze(1,2), label='true', color = 'red')
plt.plot(eval.r_dsc.squeeze(1,2), ptl_init.squeeze(1,2).detach(), label='model_i', color = 'green')
plt.plot(eval.r_dsc.squeeze(1,2), ptl_func.squeeze(1,2).detach(), label='model', color = 'blue')
#plt.plot(eval.x_spl, torch.cat((para_0, torch.tensor([0.]))).detach(), color = 'red', marker = 'o')
plt.legend()
plt.show()

#|%%--%%| <UGcCcK0clZ|Vi3de8aRVr>

print(nn.L1Loss()(eval.ptl_tr(hp['true']), ptl_md))

#|%%--%%| <Vi3de8aRVr|tgFRr8WZLe>


