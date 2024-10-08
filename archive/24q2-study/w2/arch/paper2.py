import matplotlib.pyplot as plt 
import torch, torch.nn as nn
from tqdm.auto import tqdm
import pandas as pd

# model constants
zinum = 0.01 # boundary cutoff
zfnum = 0.99 # horizon cutoff
znum = 11 # layer depth EQUALS discretisation of z
delt = (zfnum - zinum)/znum

zrange = torch.linspace(zinum, zfnum, znum)
zaxis = zrange.detach().numpy()

# numerical constants 
mu = 1 # 0 to sq12
tm = (12-mu**2)/4 # 4piT

def fp1(z, func): # first def of deriv
    return (func[z+1] - func[z])/delt

def fp2(z, func): # first def of deriv
    return (torch.log(func[z+1]) - torch.log(func[z]))*func[z]/delt

def global_for(sig, om, func):
    sr = sig.real.clone() 
    si = sig.imag.clone()
    for i in range(znum):
        z = zrange[i].item()
        fz = func[i].item()
        if i == znum - 1:
            i = i - 1

        fzp = fp1(i,func).item()
        
        print("z", z, "fz", fz, "fzp", fzp) 

        sr = sr + delt*(-fzp/fz*(sr+1/(tm*(1-z)))+2*om*sr*si+2*om/(tm*(1-z))*si-1/(tm*(1-z)**2))
        si = si + delt*(-fzp/fz*si-om*(tm**2*(1-z)**2)-2*om/(tm*(1-z))*sr+om/fz**2-mu**2*z**2/(om*fz)+om*si**2-om*sr**2)
        
        #sr = sr_t + delt*(-fzp/fz*sr_t+2*om*si_t*sr_t)
        #si = si_t + delt*(-fzp/fz*si_t+om/fz**2-mu**2*z**2/(om*fz)+om*si_t**2-om*sr_t**2)

        print("sr", sr)
        print("si", si)
    return (sr, si)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ften = nn.Parameter(torch.rand(znum))
    
    def forward(self, sig, om):
        return global_for(sig, om, self.ften)

# training sigma
## boundary sigma
ws = "~/workspace/projects/24q2-study/"
file = "mu=1_omega(0.1,1]"
dir = ws + "data/tdata/" + file + "/"

df1 = pd.read_csv(dir + "AxImsubv8.dat", sep=r"\s+", header=None)
df2 = pd.read_csv(dir + "AxResubv8.dat", sep=r"\s+", header=None)
df3 = pd.read_csv(dir + "dAxImsubv8.dat", sep=r"\s+" , header=None)
df4 = pd.read_csv(dir + "dAxResubv8.dat", sep=r"\s+", header=None)

omr = torch.tensor(df1.iloc[:, 0].values)
om = torch.complex(omr, torch.zeros_like(omr))
ax = torch.complex(torch.tensor(df2.iloc[:, 1].values),
                   torch.tensor(df1.iloc[:, 1].values))
dax = torch.complex(torch.tensor(df4.iloc[:, 1].values),
                   torch.tensor(df3.iloc[:, 1].values))

srb = dax/(1j*om*ax) - 1/(tm*(1-zinum))
#srb = torch.div(dax,1j*om*ax)

## horizon sigma
f_tr = lambda z : 1 - z**3 - mu**2*z**3/4 + mu**2*z**4/4

print("srb", srb)
print("f_tr", f_tr(zrange))
print("----")

srhr_tr, srhi_tr = global_for(srb, omr, f_tr(zrange))
print("true real : ", srhr_tr)
print("true imag : ", srhi_tr)
print("----")

# model training
## hyperparas 
epoch = 100
eta = 0.1 
reg1 = 0.01
reg2 = 0.01

model = Model()
srhr_md, srhi_md = model(srb, omr)
srhr_m0, srhi_m0 = srhr_md, srhi_md

#print(model.ften)
irange = torch.linspace(0, znum-1, znum).long()
f_e0 = model.ften[irange].detach().numpy() # epoch 0 values

optim = torch.optim.Adam(model.parameters(), lr = eta)

pbar1 = tqdm(range(epoch), desc='Training Progress', total=epoch, leave=True, position=0, colour='blue')

def learn():
    for i in range(epoch):
        srhr_md, srhi_md = model(srb, omr)
        #print("model real : ", srhr_md)
        #print("model imag : ", srhi_md)

        f_md = model.ften

        loss_1 = torch.sum(torch.sqrt((srhr_tr-srhr_md)**2) \
            + (srhi_tr-srhi_md)**2) / \
            srhr_tr.size(dim = 0)
        loss_2 = reg1 * (f_md[0] - 1)**2 
        loss_3 = reg2 / (epoch/10)**1.5 * \
            torch.sum((1/zrange[:-1]) ** 2 * \
            (f_md[1:]-f_md[:-1]) ** 2)

        loss = loss_1 + loss_2 + loss_3
        #print("loss : ", loss)

        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar1.update(1)

#learn()

f_ef = model.ften[irange].detach().numpy()

# plots
f_et = f_tr(zrange).detach().numpy()

plt.figure()
plt.plot(zaxis, f_et, color = 'red', label = 'true') # true data
plt.plot(zaxis, f_e0, color = 'blue', label = 'true') # initial data
plt.plot(zaxis, f_ef, color = 'orange', label = 'true') # model data
plt.savefig('fz.png')
plt.close()

plt.figure()
plt.plot(omr.detach().numpy(), srhr_tr.detach().numpy(), color = 'red', label = 'true')
plt.plot(omr.detach().numpy(), srhr_m0.detach().numpy(), color = 'blue', label = 'true')
plt.plot(omr.detach().numpy(), srhr_md.detach().numpy(), color = 'orange', label = 'true')
plt.savefig('srhr.png')
plt.close()

plt.figure()
plt.plot(omr.detach().numpy(), srhi_tr.detach().numpy(), color = 'red', label = 'true')
plt.plot(omr.detach().numpy(), srhi_m0.detach().numpy(), color = 'blue', label = 'true')
plt.plot(omr.detach().numpy(), srhi_md.detach().numpy(), color = 'orange', label = 'true')
plt.savefig('srhi.png')
