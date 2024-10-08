import numpy as np
import matplotlib.pyplot as plt 
import torch, torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm

# numerical constants
zb = 0.01 # boundary cutoff
zh = 0.99 # horizon cutoff
mu = 1 # set to 1 or 2; must change reading files 
pt = (12-mu**2)/4 # 4piT = 4*pi * (12-mu**2)/(16*pi)

# model constants 
z_layer = 11 # layer depth; discretisation of z
dz = (zh-zb)/z_layer

## model ranges
zrange = torch.linspace(zb, zh, z_layer)

# sigr_b training data; read files
ws = "/home/hanse/workspace/projects/archive/24q2-study/w2/"
file = "mu=1_omega(0.1,1]" # change value of mu if needed
dir = ws + "data/tdata/" + file + "/"

## change name of dat files appropriately
df1 = pd.read_csv(dir + "AxResubv8.dat", sep=r"\s+", header=None)
df2 = pd.read_csv(dir + "AxImsubv8.dat", sep=r"\s+", header=None)
df3 = pd.read_csv(dir + "dAxResubv8.dat", sep=r"\s+" , header=None)
df4 = pd.read_csv(dir + "dAxImsubv8.dat", sep=r"\s+", header=None)

omr = torch.tensor(df1.iloc[:, 0].values)
om = torch.complex(omr, torch.zeros_like(omr))

ax = torch.complex(torch.tensor(df1.iloc[:, 1].values),
                   torch.tensor(df2.iloc[:, 1].values))
dax = torch.complex(torch.tensor(df3.iloc[:, 1].values),
                   torch.tensor(df4.iloc[:, 1].values))

sigr_b = dax/(1j*om*ax) - 1/(pt*(1-zb))
# sig_b = dax(1j*om*ax)

# numerical forward 
## same function for 1) solving training data 2) model forwprop
## 1) f_tr 2) f_md
def fzp1(fz, fz2):
    return (fz2 - fz)/dz 

def fzp2(fz, fz2):
    return (np.log(fz2) - np.log(fz))*fz/dz

def numerical_forward(sig, func):
    sr = sig.real
    si = sig.imag

    for i in range(z_layer-1):
        z = zrange[i].item()
        fz = func[i].item()
        fz2 = func[i+1].item()
        fzp = fzp1(fz, fz2)

        sr = sr + dz * (
                - fzp/fz * (sr + 1/(pt * (1-z))) + 2*omr*si*sr 
                + 2*omr*si/(pt * (1-z)) - 1/(pt * (1-z)**2)
                )
        si = si + dz * (
                - fzp/fz*si - omr/(pt**2 * (1-z)**2)
                - 2*omr*sr/(pt * (1-z))
                + omr/fz**2 - mu**2 * z**2 / (omr * fz)
                + omr*si**2 - omr*sr**2
                )

    return sr, si

# sigr_h training data; numerical analysis
f_tr = lambda z : 1 - z**3 - mu**2*z**3/4 + mu**2*z**4/4

sigr_h_tr_r, sigr_h_tr_i = numerical_forward(sigr_b, f_tr(zrange))
sigr_h_tr = torch.complex(sigr_h_tr_r, sigr_h_tr_i)

# model training 
## hyperparameters
epoch = 500
eta = 0.05
reg1 = 0.01 
reg2 = 0.01

## ml model 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_md = nn.Parameter(0.2+torch.rand(z_layer))

    def forward(self, sig):
        return numerical_forward(sig.clone(), self.f_md)

model = Model()

sigr_h_md_r0, sigr_h_md_i0 = model(sigr_b)
f_md_plt0 = model.f_md.clone().detach().numpy()

loss_d = []

## model learning
def learn(model):
    optim = torch.optim.Adam(model.parameters(), lr = eta)
    
    pbar1 = tqdm(range(epoch), desc='Training Progress', total=epoch, leave=True, position=0, colour='blue')

    for _ in range(epoch):
        sigr_h_md_r, sigr_h_md_i = model(sigr_b)

        f_md = model.f_md

        loss_1 = torch.mean(torch.sqrt(
            (sigr_h_tr_r-sigr_h_md_r)**2
            + (sigr_h_tr_i-sigr_h_md_i)**2)
            )
        loss_2 = reg1 * ((f_md[0] - 1)**2 + f_md[z_layer-1]**2)
        loss_3 = reg2 / (epoch/10)**1.5 * \
            torch.sum((1/zrange[:-1]) ** 2 * \
            (f_md[1:]-f_md[:-1]) ** 2)
        
        loss = loss_1 + loss_2 + loss_3
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar1.update(1)

        loss_d.append(loss.item())

    return model

model = learn(model)

# figures
savedir = '/home/hanse/workspace/projects/archive/24q2-study/w2/'

z_plt = zrange.detach().numpy()
om_plt = omr.detach().numpy()
sigr_h_md_r, sigr_h_md_i = model(sigr_b)

## z-f(z)
f_tr_plt = f_tr(zrange).detach().numpy()
f_md_plt = model.f_md.detach().numpy()

#plt.figure()
plt.plot(z_plt, f_tr_plt, color = 'blue', label = 'true')
plt.plot(z_plt, f_md_plt0, color = 'orange', label = 'epoch 0')
plt.plot(z_plt, f_md_plt, color = 'red', label = 'epoch ' + str(epoch))

plt.xlabel('z')
plt.ylabel('f(z)')
plt.legend()

plt.savefig(savedir + 'z-fz.png')
plt.close()

## omega-sigr
sigr_h_tr_r_plt = sigr_h_tr_r.detach().numpy()
sigr_h_md_r_plt0 = sigr_h_md_r0.detach().numpy()
sigr_h_md_r_plt = sigr_h_md_r.detach().numpy()

plt.figure()
plt.plot(om_plt, sigr_h_tr_r_plt, color = 'blue', label = 'true')
plt.plot(om_plt, sigr_h_md_r_plt0, color = 'orange', label = 'epoch 0')
plt.plot(om_plt, sigr_h_md_r_plt, color = 'red', label = 'epoch ' + str(epoch))

plt.xlabel('omega')
plt.ylabel('sigma_r Re')
plt.legend()

plt.savefig(savedir + 'omega-sigr.png')
plt.close()

## omega-sigi
sigr_h_tr_i_plt = sigr_h_tr_i.detach().numpy()
sigr_h_md_i_plt0 = sigr_h_md_i0.detach().numpy()
sigr_h_md_i_plt = sigr_h_md_i.detach().numpy()

plt.figure()
plt.plot(om_plt, sigr_h_tr_i_plt, color = 'blue', label = 'true')
plt.plot(om_plt, sigr_h_md_i_plt0, color = 'orange', label = 'epoch 0')
plt.plot(om_plt, sigr_h_md_i_plt, color = 'red', label = 'epoch ' + str(epoch))

plt.xlabel('omega')
plt.ylabel('sigma_r Im')
plt.legend()

plt.savefig(savedir + 'omega-sigi.png')
plt.close()

## error
plt.figure()
plt.plot(range(epoch), loss_d)

plt.xlabel('epoch')
plt.ylabel('loss')

plt.savefig(savedir + 'loss.png')
plt.close()
