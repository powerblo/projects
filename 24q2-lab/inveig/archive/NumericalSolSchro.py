
from torch import tensor

# |%%--%%| <YSoeERReRG|HP3E5aLncK>
r"""°°°
# Basic Structure of Matrix solution of Schrodinger equation
°°°"""
# |%%--%%| <HP3E5aLncK|FQ0EN8lKwg>
r"""°°°
## Finite difference method



$$\left. \frac{d^2 f}{d x^2}\right|_{x= x_i} \approx \frac{1}{\Delta x^2} (f(x_{i-1}) -2 f(x_{i}) + f(x_{i+1}))$$

For higher terms
See [finite difference coeffficents](https://en.wikipedia.org/wiki/Finite_difference_coefficient)


°°°"""
# |%%--%%| <FQ0EN8lKwg|vC86jlVJ9y>
r"""°°°
### Conver Schrodinger equation with difference form

$$- \frac{\hbar}{2m} \frac{d^2}{d x^2} \psi  + V \psi = E \psi$$

$$- \frac{\hbar}{2m} \frac{1}{\Delta x^2} (\psi (x_{i-1}) -2 \psi (x_{i}) + \psi (x_{i+1}))  + V(x_i) \psi(x_i) = E \psi(x_i)$$

$$H \psi = E \psi$$


$$H = K +V = \begin{bmatrix}
\frac{\hbar}{m} + V(x_0) & - \frac{\hbar}{2m} & 0 & \cdots & 0 \\
- \frac{\hbar}{2m} & \frac{\hbar}{m} + V(x_1) & - \frac{\hbar}{2m} & \cdots & 0 \\
0 &- \frac{\hbar}{2m} & \frac{\hbar}{m} + V(x_1) & \cdots & 0 \\
\vdots & \vdots & \ddots & \cdots & 0\\
0   &  \cdots &   - \frac{\hbar}{2m} & \frac{\hbar}{m} + V(x_n) &  - \frac{\hbar}{2m}\\
0   &  \cdots &   0 &- \frac{\hbar}{2m} & \frac{\hbar}{m} + V(x_n)\\
\end{bmatrix}$$


$$\begin{bmatrix}
\frac{\hbar}{m} + V(x_0) & - \frac{\hbar}{2m} & 0 & \cdots & 0 \\
- \frac{\hbar}{2m} & \frac{\hbar}{m} + V(x_1) & - \frac{\hbar}{2m} & \cdots & 0 \\
0 &- \frac{\hbar}{2m} & \frac{\hbar}{m} + V(x_1) & \cdots & 0 \\
\vdots & \vdots & \ddots & \cdots & 0\\
0   &  \cdots &   - \frac{\hbar}{2m} & \frac{\hbar}{m} + V(x_n) &  - \frac{\hbar}{2m}\\
0   &  \cdots &   0 &- \frac{\hbar}{2m} & \frac{\hbar}{m} + V(x_n)\\
\end{bmatrix}
\begin{bmatrix}
\psi(x_0)\\
\psi(x_1)\\
\psi(x_2)\\
\vdots\\
\psi(x_{n-1})\\
\psi(x_n)\\
\end{bmatrix}
=E
\begin{bmatrix}
\psi(x_0)\\
\psi(x_1)\\
\psi(x_2)\\
\vdots\\
\psi(x_{n-1})\\
\psi(x_n)\\
\end{bmatrix}
$$
- $\vec{\psi}_i = \psi(x_i)$
- $V_{ii} = V(x_i)$


°°°"""
# |%%--%%| <vC86jlVJ9y|cUSUr5X23J>

import numpy as np
from scipy import linalg as scilalg

# |%%--%%| <cUSUr5X23J|NYaCSMmCJ9>

def np_off_dig(mat, i:int, vals):
    n, m = mat.shape
    if i==0:
        np.fill_diagonal(mat, vals)
    if i >0:
        np.fill_diagonal(mat[:-i, i:], vals)
    if i< 0:
        np.fill_diagonal(mat[-i:, :i], vals)
def np_get_off_dig(i, vals):
    l = len(vals) + abs(i)
    m = np.zeros((l,l))
    np_off_dig(m, i, vals)
    return m
    

# |%%--%%| <NYaCSMmCJ9|otouteXVEm>

np_get_off_dig(-2, [1,2,3,4,5])

# |%%--%%| <otouteXVEm|fWkoHrWuEV>

L = 10
dx = 0.1
N = int(L/dx)
print(N)
xline = np.arange(N)*dx - L/2
xline

# |%%--%%| <fWkoHrWuEV|QAcX0rymO1>

hm = 200 # h/m
vx = lambda x: x**2
K = np_get_off_dig(-1, -hm*np.ones(N-1)/2) + np_get_off_dig(1, -hm*np.ones(N-1)/2) + hm*np.eye(N)
V = np.eye(N)
np.fill_diagonal(V, vx(xline)) 
H = K + V

# |%%--%%| <QAcX0rymO1|jtKFGhVbPY>

result = np.linalg.eig(H)

# |%%--%%| <jtKFGhVbPY|BhNLMGDX7P>

h, v = scilalg.eigh(H)

# |%%--%%| <BhNLMGDX7P|LsL12vFCFN>

import matplotlib.pyplot as plt

# |%%--%%| <LsL12vFCFN|B3VB62Aehi>

v.shape

# |%%--%%| <B3VB62Aehi|Zl5XOzcvLV>

for i in range(10):
    e =v[:,i]
    plt.plot(xline, (e)+i, label=f"{i}th-eigen")
plt.legend()
#plt.xlim(-1, 1)

# |%%--%%| <Zl5XOzcvLV|qTu84NNl9V>

e.shape

# |%%--%%| <qTu84NNl9V|BfBAHrgyvD>
r"""°°°
## NN model
°°°"""
# |%%--%%| <BfBAHrgyvD|vJIqdXND01>

import torch
from torch import nn
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

from matplotlib import pyplot as plt

f_dim_coefs = [
    [-2,        1],
    [-2.5,      4/3,    -1/12],
    [-49/18,    3/2, 	-3/20, 	1/90]
]

def get_kinetic(N, order=0, dx=1):
    assert order <3, "Supported order is under 3"
    K = torch.zeros(N, N)
    for i, coef in enumerate(f_dim_coefs[order]):
        if i == 0:
            K = K+coef*torch.eye(N)
        else:
            arr = coef*torch.ones(N-i)
            K = K + coef*(torch.diag(arr , i) + torch.diag(arr , -i))
    return K/(dx**2)

# |%%--%%| <vJIqdXND01|YKdFSvn0Ai>

get_kinetic(5, 1)

# |%%--%%| <YKdFSvn0Ai|5keD9uO6VO>

N = 200
L = 5
xline  = torch.linspace(-L/2, L/2, N)
dx = xline[1]-xline[0]
 
coef = 0.35
vx = 100*((xline-coef*L)**2)# *(xline+coef*L)**2 #- (xline-L/2)**3 # (xline)**2

vx = vx - vx.min()
vx = vx/vx.max()

vx_fft = torch.fft.fft(vx)
V = torch.diag(vx) 
K = get_kinetic(N, 0, dx)
H = K+V

# |%%--%%| <5keD9uO6VO|42SbVuwwly>

plt.plot(xline, vx )

# |%%--%%| <42SbVuwwly|CqSLbroc2D>

def get_eighs(H):
    evals, evecs = torch.linalg.eigh(H.to(torch.float64))
    return evals, evecs

# |%%--%%| <CqSLbroc2D|YOWvSqXAMg>

evals, evec = get_eighs(H)

# |%%--%%| <YOWvSqXAMg|Hp9dAZthiX>

evals.max(), evals.min()

# |%%--%%| <Hp9dAZthiX|cHzIoCeRqo>

import matplotlib.pyplot as plt

# |%%--%%| <cHzIoCeRqo|HLf8O2T9e1>

plt.imshow(evec.real.detach())

# |%%--%%| <HLf8O2T9e1|9BvDKSW2yM>

#plt.plot(xline, 0.01*vx)
plt.plot(xline, torch.real(evec[0]).detach(), label="ground")
#plt.plot(xline, evec[1].real.detach(), label="1st")
#plt.plot(xline, evec[2].real.detach(), label="2nd")
#plt.plot(xline, evec[3].real.detach(), label="3rd")
plt.legend()

# |%%--%%| <9BvDKSW2yM|kGbVovytN7>

v_true = vx_fft

# |%%--%%| <kGbVovytN7|bwpaoMl5E8>

from torch import nn
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

# |%%--%%| <bwpaoMl5E8|D54V3jAC1X>

class VModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(VModule, self).__init__(*args, **kwargs)
        self.vx_fft = vx_fft

        self.layer = nn.Sequential(
            nn.Linear(N,    1000), nn.ReLU(),
            nn.Linear(1000,  500), nn.ReLU(),
            nn.Linear(500,  1000), nn.ReLU(),
            nn.Linear(1000, N)
        )
    def forward(self, x):
        x = x.view(N)
        return self.layer(x).view(N)


# |%%--%%| <D54V3jAC1X|U3SRr3jA3R>

module = VModule()

# |%%--%%| <U3SRr3jA3R|pZ7L1pNM7A>

xline.min(), xline.max()

# |%%--%%| <pZ7L1pNM7A|vYuLtVMImK>

float(vx[0])-eps

# |%%--%%| <vYuLtVMImK|96CgpCQS56>

vx[-1]

# |%%--%%| <96CgpCQS56|OHa6gV9uzj>

v = module(evals).detach()
eps=0.00001
plt.plot(xline, v/v.max(), label="Model Potential Before training")
plt.plot(xline, vx, label="Potential", c="r")
plt.axvline(float(xline.min()), 0.8, 10, c="r")
plt.axvline(float(xline.max()), 0.42, 10, c="r")
plt.ylim(-1, 1.5)
plt.legend()

# |%%--%%| <OHa6gV9uzj|iWPhoyygiA>

L2loss = torch.nn.MSELoss(reduction="mean")
relu = nn.ReLU()
relu_e = lambda x, e: relu(x - e)

# |%%--%%| <iWPhoyygiA|fVQxuNE4M8>

opt = torch.optim.Adam(module.parameters(), lr=0.2)

# |%%--%%| <fVQxuNE4M8|Uap0dRAJZZ>

epoches = 16000
for epoch in range(epoches):

    v = module(evals)
    v = v-v.min()
    v = v/v.max()
    V = torch.diag(v)
    H_m = K+V
    evals_m, evec = get_eighs(H_m)
    max_diff = 0.1

    #loss_symmetry = L2loss(v, torch.flip(v, [0]))
    loss_dif = (1000/(epoch+2))*torch.max(relu_e(torch.diff(evals_m), max_diff))
    #print(loss_dif.item())
    # 양끝 1 whrjs
    loss_eigen = 200*L2loss(evals_m, evals)
    loss_v0 = (v[0]-1)**2
    
    loss = loss_eigen + loss_v0 + loss_dif #+ loss_symmetry
    print(f"{loss_eigen/loss:.3}|{loss_dif/loss:.3}|{0/loss:.3}")
    print(f"{epoch}:{loss.item()}")
    #loss = 200*L2loss(evals_m, evals) + (torch.max(v)-1)**2 + loss_dif
    

    opt.zero_grad()
    loss.backward()
    opt.step()

# |%%--%%| <Uap0dRAJZZ|i7Ck0oD67a>

v_model = module(evals).detach()
v_model = v_model-v_model.min()
v_model = v_model/v_model.max()

plt.plot(xline, torch.flip(v.detach(), [0]), label="Tranined")
plt.plot(xline, vx, label="True")
plt.legend()

# |%%--%%| <i7Ck0oD67a|ecIRUwkZ54>

v_model = module(evals).detach()
v_model = v_model-v_model.min()
v_model = v_model/v_model.max()

plt.plot(xline, v.detach(), label="Tranined")
plt.plot(xline, vx, label="True")
plt.legend()

# |%%--%%| <ecIRUwkZ54|lN0HWLZ70B>

v_model = module(evals).detach()
v_model = v_model-v_model.min()
v_model = v_model/v_model.max()

plt.plot(xline, v.detach(), label="Tranined")
plt.plot(xline, vx, label="True")
plt.legend()

# |%%--%%| <lN0HWLZ70B|P1OFoaiaBI>

v_model = module(evals).detach()
v_model = v_model-v_model.min()
v_model = v_model/v_model.max()

plt.plot(xline, v.detach(), label="Tranined")
plt.plot(xline, vx, label="True")
plt.legend()

# |%%--%%| <P1OFoaiaBI|K1S2IZyGXJ>

plt.plot(v.detach())
plt.plot(torch.flip(v, [0]).detach())


# |%%--%%| <K1S2IZyGXJ|aXbDcPCyib>

plt.stem(evals)

# |%%--%%| <aXbDcPCyib|zKPVjgtmJF>

plt.stem(evals_m.detach())

# |%%--%%| <zKPVjgtmJF|5uE8NLrLWM>


