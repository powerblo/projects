import matplotlib.pyplot as plt 
import torch, torch.nn as nn
import pandas as pd 

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
ws = "~/workspace/projects/24q2-study/"
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
def numerical_forward(sig, func):
    sr = sig.real
    si = sig.imag
    print(sig)
    print("real : ", sr)
    print("imag : ", si)

    for i in range(z_layer-1):
        print("----")
        z = zrange[i].item()
        print("index : ", i, "z : ", z)
        fz = func[i].item()
        fz2 = func[i+1].item()
        print("f(z) : ", fz, "f(z+dz) : ", fz2)
        fzp = (fz2 - fz) / dz 
        print("f'(z) : ", fzp)

        sr = sr + dz * (
                - fzp/fz * ( sr + 1/(pt * (1-z)) ) + 2*omr*si*sr 
                + 2*omr*si/(pt * (1-z)) - 1/(pt * (1-z)**2)
                )
        si = si + dz * (
                - fzp/fz*si - omr/(pt**2 * (1-z)**2)
                - 2*omr*sr/(pt * (1-z))
                + omr/fz**2 - mu**2 * z**2 / (omr * fz)
                + omr*si**2 - omr*sr**2
                )
        print("real : ", sr)
        print("imag : ", si)

    return sr, si

# sigr_h training data; numerical analysis
f_tr = lambda z : 1 - z**3 - mu**2*z**3/4 + mu**2*z**4/4

print("====")
print("true numerical")
sigr_h_tr_r, sigr_h_tr_i = numerical_forward(sigr_b, f_tr(zrange))
sigr_h_tr = torch.complex(sigr_h_tr_r, sigr_h_tr_i)
#print(sigr_h_tr)

# model training 
## hyperparameters
epoch = 1000
eta = 0.05
reg1 = 0.01 
reg2 = 0.01

## ml model 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_md = nn.Parameter(torch.rand(z_layer))

    def forward(self, sig):
        return numerical_forward(sig.clone(), self.f_md)

model = Model()
print("====")
print("model numerical")
sigr_h_md_r, sigr_h_md_i = model(sigr_b)

# figures
z_plt = zrange.detach().numpy()
om_plt = omr.detach().numpy()

## z-f(z)
f_tr_plt = f_tr(zrange).detach().numpy()
f_md_plt = model.f_md.detach().numpy()

plt.figure()
plt.plot(z_plt, f_tr_plt, color = 'blue', label = 'true')
plt.plot(z_plt, f_md_plt, color = 'red', label = 'model')

plt.savefig('z-fz.png')
plt.close()

## omega-sigr
sigr_h_tr_r_plt = sigr_h_tr_r.detach().numpy()
sigr_h_md_r_plt = sigr_h_md_r.detach().numpy()

plt.figure()
plt.plot(om_plt, sigr_h_tr_r_plt, color = 'blue', label = 'true')
plt.plot(om_plt, sigr_h_md_r_plt, color = 'red', label = 'model')

plt.savefig('omega-sigr.png')
plt.close()

## omega-sigi
sigr_h_tr_i_plt = sigr_h_tr_i.detach().numpy()
sigr_h_md_i_plt = sigr_h_md_i.detach().numpy()

plt.figure()
plt.plot(om_plt, sigr_h_tr_i_plt, color = 'blue', label = 'true')
plt.plot(om_plt, sigr_h_md_i_plt, color = 'red', label = 'model')

plt.savefig('omega-sigi.png')
plt.close()
