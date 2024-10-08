import numpy as np
import matplotlib.pyplot as plt 
import torch, torch.nn as nn
import pandas as pd
from tqdm.auto import tqdm

# numerical constants

# model constants

# s, chi_2 training data; read files
#temp_s, s_tr = [0], [0]
temp_s = torch.linspace(0.13, 0.3, 35)
s_tr = torch.tensor([2.77, 3.13, 3.58, 4.11, 4.7, 5.34, 6, 6.66, 
7.32, 7.94, 8.53, 9.08, 9.6, 10.07, 10.5, 10.9, 11.27, 11.61, 11.92, 
12.21, 12.49, 12.75, 13, 13.24, 13.47, 13.68, 13.89, 14.09, 14.28, 
14.46, 14.53, 14.8, 14.95, 15.11, 15.25]) * temp_s**3
temp_chi, chi_tr = [0], [0]

# model training
## hyperparameters
epoch = 2000
eta = 0.01

layer_s = [1, 64, 128, 64, 1]

layer_p = 1
for k in layer_s:
    layer_p = layer_p * np.sqrt(k)

## ml model 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []

        for i in range(np.array(layer_s).size - 1):
            layers.append(nn.Linear(layer_s[i], layer_s[i+1]))

        self.layers = nn.ModuleList(layers)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight.data, a = 0.0, b = 1.0/layer_p)
                nn.init.uniform_(m.bias.data)

        self.act = nn.Sigmoid()

    def forward(self, z):
        data = z.view(-1, 1)
        for i in range(np.array(layer_s).size - 1):
            data = self.act(self.layers[i](data))
        
        return data.view(-1)

s_model = Model()
#chi_model = Model()

s_0 = s_model(temp_s) / temp_s**3
#chi_0 = chi_model(temp)

## model learning
loss_s_d, loss_chi_d = [], []

def learn(model, temp, tr, loss_d):
    optim = torch.optim.Adam(model.parameters(), lr = eta)
    
    pbar1 = tqdm(range(epoch), desc='Training Progress', total=epoch, leave=True, position=0, colour='blue')

    for _ in range(epoch):
        m = model(temp)

        loss = nn.L1Loss()(m, tr)

        loss_d.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()

        pbar1.update(1)

    return model

s_model = learn(s_model, temp_s, s_tr, loss_s_d)
#chi_model = learn(chi_model, chi_tr, loss_chi_d)

s_1 = s_model(temp_s) / temp_s**3
#chi_1 = chi_model(temp)

plt.plot(temp_s,(s_tr/temp_s**3).detach().numpy(), label='s_tr')
#plt.plot(s_0.detach().numpy(), label='s_0')
plt.plot(temp_s,s_1.detach().numpy(), label='s_1')
plt.legend()
#plt.show()
plt.close()

plt.plot(loss_s_d)
#plt.show()
plt.close()

# analytic computation
## invert temp_zh into zh_temp
## para : [a, b, d, c, k, g]
#def temp_zh(zh, para):
#    a, b, d, c, k, g = para[:6]
#    return ( torch.abs(2*c*zh - 2*a*d*zh / (a*zh**2 + 1)
#        - 4*b*d*zh**3 / (b*zh**4 + 1))
#        * (torch.exp(c * zh**2 - d * torch.log(a*zh**2 + 1)
#        - d * torch.log(b*zh**4 + 1) + k)) ) / 4 * torch.pi

def expa(zh, para): # zh^3/exp(3A(zh))
    a, b, d, c, k, g = para[:6]
    return zh**3*((a*zh**2+1)*(b*zh**4+1))**(-3*d)

def temp_zh(zh, para):
    zh_m = zh.max().item()
    zh_lin = torch.linspace(0, zh_m, 1000)
    
    integ_val = torch.zeros_like(zh)

    for i, zh_i in enumerate(zh):
        integ_val[i] = torch.trapz(expa(zh_lin[zh_lin <= zh_i], para), zh_lin[zh_lin <= zh_i])
    
    return (expa(zh, para)/(4*torch.pi*integ_val))

def s_temp(zh, temp, para):
    a, b, d, c, k, g = para[:6]
    return 1/(4*g*expa(zh, para))

def chi_temp(temp, para):
    a, b, d, c, k, g = para[:6]
    zh = zh_temp(temp, para)
    return (c*np.exp(k))/(8*torch.pi*g*temp**2*(1-torch.exp(-c*zh**2)))

class Fitting(nn.Module):
    def __init__(self):
        super().__init__()
        self.paras = nn.Parameter(torch.rand(6))
        self.paras.data[2] = -torch.abs(self.paras.data[2])
        self.paras.data[3] = -torch.abs(self.paras.data[3])
        #self.paras = nn.Parameter(torch.tensor([0.204, 0.013, -0.264, -0.173, -0.824, 0.4]))

    def forward(self, zh, temp):
        return s_temp(zh, temp, self.paras)

def fit(model):
    optim = torch.optim.SGD(model.parameters(), lr = eta)
    for k in range(fit_epoch):
        temp_arr = temp_zh(zh_arr, model.paras)
        s_2t = s_model(temp_arr)
        s_2 = model(zh_arr, temp_arr)
        #chi_2 = chi_temp(temp)

        print("----", k)

        print("s_2 : ", s_2)
        print("s_2t : ", s_2t)
        print("paras : ", model.paras)

        loss = (nn.L1Loss()(s_2, s_2t) + torch.relu(-model.paras[0]) + torch.relu(-model.paras[1]))

        print("loss : ", loss)

        optim.zero_grad()
        loss.backward(retain_graph = True)
        optim.step()
    
    return model

zh_i, zh_f, zh_n = 0.2, 2, 100
zh_arr = torch.linspace(zh_i, zh_f, zh_n)

fit_epoch = 5

model = Fitting()

model = fit(model)

#print(model.paras)

plt.plot(temp_s,(s_tr/temp_s**3).detach().numpy(), label='s_tr')

temp_arr = temp_zh(zh_arr, model.paras)

#print(model.paras, temp_arr)
plt.plot(temp_arr.detach().numpy(), (s_model(temp_arr)/temp_arr**3).detach().numpy(), label='s_1')
plt.plot(temp_arr.detach().numpy(), (model(zh_arr, temp_arr)/temp_arr**3).detach().numpy(), label='s_2')
plt.legend()
plt.show()
#plt.close()