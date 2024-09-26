r"""°°°
<양자역학:해밀토니안을 구해보자>
°°°"""
# |%%--%%| <C80bqgYcs2|FOPxiiukXR>

import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np

# |%%--%%| <FOPxiiukXR|rUPIXsitYJ>

a=5
ep=0.05
end=(a/ep)+1

hbar=1
m=1

#x=[0:a/(N-1):a]
x=torch.linspace(0, a, int(end))#x=[0:ep:a]
N=len(x)
print(x.size())

identity_matrix = torch.eye(N)  
off_diag = torch.ones(N-1) 

#운동량 연산자(x-bais)
K = (1/ep**2) * ((-hbar**2) / (2 * m)) * (-2 * identity_matrix + torch.diag(off_diag, 1) + torch.diag(off_diag, -1))
#K=K.float()
print(K.size())

#Infinite well Potential(x-bais)
V_infi=torch.zeros(N,N)
i=1
V_infi[0:i,0:i]=1e3
V_infi[N-i:,N-i:]=1e3 
print(V_infi.size())

#Hamiltonian
H = K+V_infi

# |%%--%%| <rUPIXsitYJ|MUQuGCVgKR>

V_infi_flat=torch.diag(V_infi)

plt.figure()
plt.plot(x, V_infi_flat, 'c--')
plt.xlim(0,a)
#plt.ylim(-1,1)
plt.show()

# |%%--%%| <MUQuGCVgKR|KSlLO3M5zF>

#true value
def eigen(Ht):
    eigenvals , eigenvecs = torch.linalg.eigh(Ht)
    return eigenvals , eigenvecs

eigenvals_tr , eigenvecs_tr=eigen(H)



# |%%--%%| <KSlLO3M5zF|i4Nzlp5GV7>

for i in range(0,3):
    plt.plot(x, eigenvecs_tr[:,i])
plt.plot(x, 1e-15*V_infi_flat, "r--")

# |%%--%%| <i4Nzlp5GV7|q7Y5FMiBcO>

eigenvals_tr

# |%%--%%| <q7Y5FMiBcO|mHnfKPYJyS>

eigenvals_tr/eigenvals_tr[0]

# |%%--%%| <mHnfKPYJyS|S7r5MwfwVc>

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1=nn.Linear(2,10)
        self.L2=nn.Linear(10,1)
        self.act= nn.ReLU() 

    def forward(self, x, eigenvalss):
        data = torch.cat((x.view(-1, 1), eigenvalss.view(-1, 1)), dim=1)
        layer = self.act(self.L1(data))
        output = self.L2(layer)
        return output.view(-1)


model=Model()
V_model=model(x,eigenvals_tr) #output
Vdiag_model=torch.diag(V_model) #diagonalized output


# |%%--%%| <S7r5MwfwVc|vavtjGp5Ci>

'''
a=torch.tensor([1,2,3])
b=torch.tensor([[2],[2],[3]])
b.size()
a1=torch.diag(a)
b1=torch.diag(b)
'''

# |%%--%%| <vavtjGp5Ci|VN36G2to6b>

#epoch 돌리기

optimizer = torch.optim.Adam( model.parameters(),lr=0.1)


for j in range(500):

    V_model=model(x,eigenvals_tr)
    Vdiag_model=torch.diag(V_model)
    H_model=K+Vdiag_model
    
    eigenvals_model, eigenvecs_model = eigen(H_model)

    loss = torch.mean(torch.abs(eigenvals_tr - eigenvals_model))
    print(f"Iteration {j}, Loss: {loss.item()}")

    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# |%%--%%| <VN36G2to6b|FVVJXxc8N7>

V_flat_model=torch.diag(Vdiag_model.detach())

plt.figure()
plt.plot(x, V_flat_model, 'c--', color='r')
plt.plot(x, V_infi_flat, 'c--', color='b')
plt.xlim(-1,6)
plt.ylim(-10,200)
plt.show()

