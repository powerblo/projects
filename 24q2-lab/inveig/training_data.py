
import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt
import scipy.sparse as sparse
from tqdm.auto import tqdm

def seq_mlp(init, mlp, fin, act):
    modules = [nn.Linear(init, mlp[0]), act]
    for i in range(len(mlp) - 1):
        modules.append(nn.Linear(mlp[i], mlp[i+1]))
        modules.append(act)

    modules.append(nn.Linear(mlp[-1], fin)) #self.spl for spline

    return modules

# |%%--%%| <FPKx04IwJu|uVysTeoK8k>

class EvalEig(nn.Module):
    def __init__(self, eval_para):
        super().__init__()
        self.bd = eval_para['batch_dim']

    def set_rdsc(self, xm, xn, p_num):
        self.xn = xn
        self.xm = xm
        self.pn = p_num
    
    def mesh_ptl(self, posx, posy): # input shape (bd, p_num)
        X, Y = np.meshgrid(np.linspace(-self.xm, self.xm, self.xn),
                            np.linspace(-self.xm, self.xm, self.xn), indexing = 'ij')
        X_broad, Y_broad = X[np.newaxis,np.newaxis,:,:], Y[np.newaxis,np.newaxis,:,:]
        posx_broad, posy_broad = posx[:,:,np.newaxis,np.newaxis], posy[:,:,np.newaxis,np.newaxis]

        dist = np.sqrt((X_broad-posx_broad)**2+(Y_broad-posy_broad)**2)
        dist[dist==0] = np.finfo(float).eps

        ptl = np.sum(-1/dist,axis=1)
        return ptl # shape (bd, xn, xn)
    
    def mesh_hml(self, term_ptl):
        dx = 2*self.xm/(self.xn-1)
        diag = [np.full(self.xn, -2/dx**2), np.full(self.xn-1, 1/dx**2), np.full(self.xn-1, 1/dx**2)]

        term_kin_partial = sparse.diags(diag, [0,-1,1], shape=(self.xn,self.xn))
        term_kin = sparse.kron(sparse.identity(self.xn), term_kin_partial) + \
            sparse.kron(term_kin_partial, sparse.identity(self.xn))
        term_hml = term_kin + sparse.diags(term_ptl.ravel(), 0)
        
        return term_hml
    
    def init_evl(self):
        posx = np.random.uniform(-self.xm, self.xm, size=(self.bd, self.pn))/10
        posy = np.random.uniform(-self.xm, self.xm, size=(self.bd, self.pn))/10

        evl = np.zeros((self.bd, 6)) # p_num as cutoff for number of smallest evls obtained? fix as 6?
        mesh_ptl = self.mesh_ptl(posx, posy)

        pbar = tqdm(range(self.bd), desc='Progress', total=self.bd, leave = True, position=0, colour='blue')

        for i in range(self.bd):
            mesh_hml = self.mesh_hml(mesh_ptl[i])
            evl_i, _ = sparse.linalg.eigsh(mesh_hml, which = 'SM')
            evl[i] = evl_i

            pbar.update()
        
        return posx, posy, evl

    def forward(self):
        posx_tr, posy_tr, evl_tr = self.init_evl()

        return posx_tr, posy_tr, evl_tr

class InvEig(EvalEig):
    def __init__(self, eval_para, model_para):
        super().__init__(eval_para)
        self.mlp_shape = model_para['mlp']

    def set_rdsc(self, xm, xn, p_num):
        self.xn = xn
        self.xm = xm
        self.pn = p_num

        # initialise model
        #self.ptl = nn.Parameter(torch.rand(self.batch_dim, self.rn-1)) # random parameters
        modules = seq_mlp(init = 6, mlp = self.mlp_shape, fin = int(self.pn*(self.pn-1)/2), act = nn.ReLU())
        self.mlp = nn.Sequential(*modules)
    
    def dist_tsor(self, posx, posy):
        diffx = posx.unsqueeze(2) - posx.unsqueeze(1)
        diffy = posy.unsqueeze(2) - posy.unsqueeze(1)
        dist = diffx**2 + diffy**2

        upptri_ind = torch.triu_indices(row=diffx.shape[1],col=diffx.shape[1],offset=1)
        upptri_val = dist[:, upptri_ind[0], upptri_ind[1]]
        val, _ = torch.sort(upptri_val)

        return val/self.xm**2

    def forward(self, evl):
        #pos = self.mlp(evl)
        #posx_md, posy_md = pos[:,:self.pn], pos[:,self.pn:]
        #val_md = self.dist_tsor(posx_md, posy_md)
        posx_md, posy_md = None, None 
        val_md = self.mlp(evl)

        return posx_md, posy_md, val_md


# |%%--%%| <uVysTeoK8k|xXhJasuNef>

eval_para = {
        # model specifics
        'precision' : 64, # 32 or 64 bit
        'batch_dim' : 1000
        }

model_para = {
        # model
        'mlp' : [1000, 1000, 1000],

        # training
        'epoch' : 5000,
        'lr' : 1e-2,

        # loss regularisation
        'reg1' : 1e-1, # V(0) sign
        'reg2' : 1, # V -> 0 as r -> infty
        
        }

eval = EvalEig(eval_para)
eval.set_rdsc(xm = 1e4, xn = 100, p_num = 10)

# |%%--%%| <xXhJasuNef|S7Cvn4f1LQ>

#eval_grid = [[800], \
#    [10000], \
#        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] # rm, rn, para_1
#for midx in itertools.product(*eval_grid):
#for midx in zip(*eval_grid):
    #eval.set_rdsc(midx[0], midx[1])
    #ptl_tr = eval.fixed_tr(midx[2], "coulomb")
    #evl_scl_tr = eval.dsc_eigs(ptl_tr)
    #evl_tr = evl_scl_tr[:,:,:eval.evl_cutoff(evl_scl_tr)]
#    ptl_tr, evl_tr = eval(midx[0], midx[1], midx[2], "coulomb")
#    factor = torch.mean(1/evl_tr, dim = 0)
#    print(factor[0])
#    print(midx, nn.L1Loss()(factor[0],torch.arange(1,factor.shape[1]+1)**2), evl_tr[0,0,0])


# |%%--%%| <S7Cvn4f1LQ|dqqtMw4W8d>

#posx_tr, posy_tr, evl_tr = eval()

# |%%--%%| <dqqtMw4W8d|lnmGKmDP1w>

#import pickle
#with open("posx_tr.data", "wb") as fw:
#    pickle.dump(posx_tr, fw)
#with open("posy_tr.data", "wb") as fw:
#    pickle.dump(posy_tr, fw)
#with open("evl_tr.data", "wb") as fw:
#    pickle.dump(evl_tr, fw)

# |%%--%%| <lnmGKmDP1w|vaDwRq94bB>

import pickle
with open("posx_tr.data", "rb") as fr:
    posx_tr = torch.from_numpy(pickle.load(fr)).to(dtype = torch.float32)
with open("posy_tr.data", "rb") as fr:
    posy_tr = torch.from_numpy(pickle.load(fr)).to(dtype = torch.float32)
with open("evl_tr.data", "rb") as fr:
    evl_tr = torch.from_numpy(pickle.load(fr)).to(dtype = torch.float32)

# |%%--%%| <vaDwRq94bB|tIwq7WLkxg>

model = InvEig(eval_para, model_para)
model.set_rdsc(xm = 1e4, xn = 1000, p_num = 10)
#model.load_state_dict(torch.load('1.pth'))

optimiser = torch.optim.Adam(model.parameters(), lr = model_para['lr'])
epochs = model_para['epoch']
pbar = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')
loss_list = [[]]

val_tr = model.dist_tsor(posx_tr, posy_tr).to(dtype = torch.float32)

#|%%--%%| <tIwq7WLkxg|hCHoYCR6v7>

for e in range(epochs):
    #with torch.autograd.detect_anomaly():
    posx_md, posy_md, val_md = model(evl_tr)
    if e == 0:
        val_init = val_md

    loss = nn.L1Loss()(val_tr, val_md)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    pbar.update()


# |%%--%%| <hCHoYCR6v7|e9rRdHJzmQ>

torch.save(model.state_dict(), f"{eval_para['batch_dim']}.pth")


#|%%--%%| <e9rRdHJzmQ|juRRVNYsNY>

print(val_tr)
print(val_init)
print(val_md)
print(loss)

