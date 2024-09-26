r"""°°°
<양자역학:해밀토니안을 구해보자>
°°°"""
# |%%--%%| <j06Uic1ww0|QxIYNYhnHG>

import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np

# |%%--%%| <QxIYNYhnHG|KoMKR5zjfw>

#x-axis range
start=-5
end=5
ep=0.05
num=((end-start)/ep)+1

#constants
hbar=1
m=1
w=1

#x=[0:a/(N-1):a]
x=torch.linspace(start, end, int(num))#x=[0:ep:a]
N=len(x)
print(x.size())

identity_matrix = torch.eye(N)  
off_diag = torch.ones(N-1) 

#운동량 연산자(x-bais)
K = (1/ep**2) * ((-hbar**2) / (2 * m)) * (-2 * identity_matrix + torch.diag(off_diag, 1) + torch.diag(off_diag, -1))
print(K.size())

#HO Potential(x-bais)
V_ho = (1/2)*m*w**2*torch.diag(x)**2
V_ho[0,0]=1e3
V_ho[N-1,N-1]=1e3
print(V_ho.size())
V_ho_flat = (1/2)*m*w**2*x**2

#Hamiltonian
H = K+V_ho

# |%%--%%| <KoMKR5zjfw|5mkXBWUUl8>

plt.figure()
plt.plot(x, V_ho_flat, 'c--')
plt.xlim(start,end)
#plt.ylim(-1,1)
plt.show()

# |%%--%%| <5mkXBWUUl8|PtR8wZCHaT>

#true value
def eigen(Ht):
    eigenvals , eigenvecs = torch.linalg.eigh(Ht)
    return eigenvals , eigenvecs

eigenvals_tr , eigenvecs_tr=eigen(H)



# |%%--%%| <PtR8wZCHaT|20dSp5P3vu>

plt.figure()
for i in range(0,3):
    plt.plot(x, eigenvecs_tr[:,i])

plt.ylim(-0.3,0.8)
plt.plot(x, 1e-1*V_ho_flat, "r--")
plt.show()

#plt.yticks(np.arange(-0.4,0.5,0.1))


# |%%--%%| <20dSp5P3vu|8o4XPYAkyS>

eigenvals_tr

# |%%--%%| <8o4XPYAkyS|QViPYMJY0x>

eigenvals_tr/eigenvals_tr[0]

# |%%--%%| <QViPYMJY0x|WNm7ey865j>

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1=nn.Linear(2,40)
        self.L2=nn.Linear(40,1)
        self.act= nn.ReLU() 

    def forward(self, x, eigenvalss):
        data = torch.cat((x.view(-1, 1), eigenvalss.view(-1, 1)), dim=1)
        layer = self.act(self.L1(data))
        output = self.L2(layer)
        return output.view(-1)


model=Model()
V_model=model(x,eigenvals_tr) #output
Vdiag_model=torch.diag(V_model) #diagonalized output


# |%%--%%| <WNm7ey865j|areDE11Amk>

#epoch 돌리기

optimizer = torch.optim.Adam( model.parameters(),lr=0.1)


for j in range(1000):

    V_model=model(x,eigenvals_tr)
    Vdiag_model=torch.diag(V_model)
    H_model=K+Vdiag_model
    
    eigenvals_model, eigenvecs_model = eigen(H_model)

    loss = torch.mean(torch.abs(eigenvals_tr - eigenvals_model))
    print(f"Iteration {j}, Loss: {loss.item()}")

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# |%%--%%| <areDE11Amk|0ejrFYDV2p>

V_flat_model=torch.diag(Vdiag_model.detach())

plt.figure()
plt.plot(x, V_ho_flat, 'c--', color='r')
plt.plot(x, V_flat_model, 'c--', color='b')
#plt.xlim(-1,6)
plt.ylim(-10,30)
plt.show()


# |%%--%%| <0ejrFYDV2p|46uWRcKvBi>


