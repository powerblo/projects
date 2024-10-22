import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CommonModule(nn.Module):
    def __init__(self, node_dim, batch_dim, device, unif_dim = 128, mlp_dim = 512, head_dim = 16, head_num = 8):
        super(CommonModule, self).__init__()
        # external model specs
        self.node_dim = node_dim
        # internal model specs
        self.unif_dim = unif_dim
        self.mlp_dim = mlp_dim
        self.batch_dim = batch_dim
        # mha specs
        self.head_dim = head_dim # should use key_dim and vec_dim separately
        self.head_num = head_num

        self.device = device
        
        self.batch_arr = torch.arange(self.batch_dim, device = self.device)

    def compat(self, Wq, Wk, Wv, vec):
        qv, kv, vv = ( # vectors : batch dim x node dim x head num x head dim
        torch.einsum('xij,abx->abij', Wq, vec),
        torch.einsum('xij,abx->abij', Wk, vec),
        torch.einsum('xij,abx->abij', Wv, vec))
        
        compat = torch.einsum('aibx,ajbx->aijb',qv,kv)/np.sqrt(self.head_dim) # compat : batch dim x node dim x node dim x head num

        return compat, vv

    def attention(self, compat, vv, Wo):
        weight = F.softmax(compat, dim = 2) # weight : batch dim x node dim x node dim x head num
        received_vec = torch.einsum('aixb,axbj->aibj', weight, vv) # recvec : batch dim x node dim x head num x head dim
        mha = torch.einsum('xyj,aixy->aij', Wo, received_vec) # mha : batch dim x node dim x unif dim
        return mha

    def GenerateRandomGraph(self):
        adj_matr = torch.zeros((self.node_dim, self.node_dim), device = self.device)

class Encoder(CommonModule):
    def __init__(self, encoder_layers, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encoder_layers = encoder_layers

        # Paramters
        self.init_embed = nn.Linear(self.node_dim, self.unif_dim, device = self.device)

        self.mha_mlp = nn.ModuleList([nn.Sequential(
                nn.Linear(self.unif_dim, self.mlp_dim, device = self.device),
                nn.ReLU(),
                nn.Linear(self.mlp_dim, self.unif_dim, device = self.device)) for _ in range(self.encoder_layers)])

        ## Market encoder MHA
        self.qv_p = [nn.Parameter(torch.rand((self.unif_dim, self.head_num, self.head_dim), device = self.device)) 
                                   for _ in range(self.encoder_layers)]
        self.kv_p = [nn.Parameter(torch.rand((self.unif_dim, self.head_num, self.head_dim), device = self.device)) 
                                   for _ in range(self.encoder_layers)]
        self.vv_p = [nn.Parameter(torch.rand((self.unif_dim, self.head_num, self.head_dim), device = self.device)) 
                                   for _ in range(self.encoder_layers)]
        self.ov_p = [nn.Parameter(torch.rand((self.head_num, self.head_dim, self.unif_dim), device = self.device)) 
                                   for _ in range(self.encoder_layers)]
        
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(self.node_dim, device = self.device) for _ in range(self.encoder_layers)])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(self.node_dim, device = self.device) for _ in range(self.encoder_layers)])

    def forward(self, adj_matr): # adj matrix : batch dim x node dim x node dim
        hvec_l = self.init_embed(adj_matr) # embedding : batch dim x node dim x unif dim
        for l in range(self.encoder_layers):
            compat, vv_l = self.compat(self.qv_p[l], self.kv_p[l], self.vv_p[l], hvec_l)
            adj_matr_n = torch.where(adj_matr==0., -1e10, adj_matr)
            compat_mask = compat*(adj_matr_n.unsqueeze(-1)) # mask non adjacent nodes 
            mha_l = self.attention(compat_mask, vv_l, self.ov_p[l])
            
            hhat_l = self.bn1[l](hvec_l + mha_l)
            hvec_l = self.bn2[l](hhat_l + self.mha_mlp[l](hhat_l)) # hvec : batch dim x node dim x unif dim

        hbar_l = torch.sum(hvec_l, dim = 1)/(1+self.node_dim) # hbar : batch dim x unif dim

        return hvec_l, hbar_l

class PathModule(CommonModule):
    def __init__(self, clipp, **kwargs):
        super(PathModule, self).__init__(**kwargs)
        self.clipp = clipp

        # Parameters : Decoder LSTM + MHA
        self.qv_p = nn.Parameter(torch.rand((3*self.unif_dim, self.head_num, self.head_dim), device = self.device)) 
        self.kv_p = nn.Parameter(torch.rand((self.unif_dim, self.head_num, self.head_dim), device = self.device)) 
        self.vv_p = nn.Parameter(torch.rand((self.unif_dim, self.head_num, self.head_dim), device = self.device)) 
        self.ov_p = nn.Parameter(torch.rand((self.head_num, self.head_dim, self.unif_dim), device = self.device)) 

        self.qvf_p = nn.Linear(self.unif_dim, self.unif_dim, device = self.device)
        self.kvf_p = nn.Linear(self.unif_dim, self.unif_dim, device = self.device)
        
        self.vec_1, self.vec_f = (nn.Parameter(torch.rand(1,self.batch_dim,self.unif_dim, device=self.device)), nn.Parameter(torch.rand(1,self.batch_dim,self.unif_dim, device = self.device)))

    def MaskAtt(self, compat, route, adj_matr, obj_list, hml):
        compat = compat.reshape(compat.shape[0],compat.shape[1],-1)

        avail = obj_list[route.transpose(0,1)].reshape(self.batch_dim,-1)
        matches_a = (hml.unsqueeze(2) == avail.unsqueeze(1))
        matches = torch.sum(matches_a, dim = 2, dtype=torch.bool).unsqueeze(-1)
        
        if torch.isin(0,hml):
            matches[:,0,0] = 1

        check = matches.all(dim=1)# * adj_matr.squeeze(0)[route[-1],0].unsqueeze(-1).to(torch.bool)
        check_demand = torch.concat((check,~check*torch.ones(self.batch_dim, self.node_dim-1,dtype=torch.bool,device=self.device)),dim=1).unsqueeze(-1).expand(compat.shape)
        compat_1 = torch.where(check_demand, compat, -1e10*torch.ones_like(compat))

        #print(adj_matr[0,route[-1]][0,:10], torch.sum(adj_matr[0,route[-1]][0]))

        check2 = adj_matr[0,route[-1]].unsqueeze(-1).to(torch.bool)
        compat_2 = torch.where(check2, compat_1, -1e30*torch.ones_like(compat_1))

        #print(compat_2[0,:10,0])

        return compat_2

    def forward(self, hvec, hbar, adj_matr, obj_list, hml, baseline = False):
        route = torch.zeros(1, self.batch_dim , dtype = torch.int, device = self.device)
        final_log_prob = torch.zeros(self.batch_dim, device = self.device)
        length = torch.zeros(self.batch_dim, device = self.device)

        hbar_b = hbar.view(1,1,-1).expand(1,self.batch_dim,-1)

        # run decoder
        max_iter = 7*5
        for t in range(2*max_iter):
            if route.shape[0] == 1:
                vec_concat = torch.concat((hbar_b, self.vec_1, self.vec_f), dim = 2).squeeze(0)
            else:
                vec_concat = torch.concat((hbar_b, hvec[:,route[t-1]], hvec[:,route[1]]), dim = 2).squeeze(0)

            qv, kv, vv = ( 
            torch.einsum('xij,ax->aij', self.qv_p, vec_concat), # batch dim x head num x head dim
            torch.einsum('xij,abx->abij', self.kv_p, hvec), # batch dim x node dim x head num x head dim
            torch.einsum('xij,abx->abij', self.vv_p, hvec))

            compat = torch.einsum('abx,aibx->aib',qv,kv)/np.sqrt(self.head_dim) # batch dim x node dim x head num
            compat_mask = self.MaskAtt(compat, route, adj_matr, obj_list, hml)
            weight = F.softmax(compat_mask, dim = 1) # weight : batch dim x node dim x head num
            # (c) x node dim  X   
            received_vec = torch.einsum('axb,axbi->abi', weight, vv) # recvec : batch dim x head num x head dim
            hvec_c = torch.einsum('xyj,ixy->ij', self.ov_p, received_vec) # hvec_c : batch dim x unif dim

            qvf = self.qvf_p(hvec_c) # qvf : batch dim x head dim = unif dim (1 head)
            kvf = self.kvf_p(hvec) # hvf : batch dim x node dim x head dim = unif dim

            compatf = self.clipp * torch.tanh(torch.einsum('ax,aix->ai', qvf, kvf)/np.sqrt(self.unif_dim)) # compatf : batch dim x node dim
            compatf_mask = self.MaskAtt(compatf, route, adj_matr, obj_list, hml).squeeze(-1)
            
            policy = F.softmax(compatf_mask, dim = 1)

            #print(torch.sum(adj_matr[0,route[-1,0]]), policy[0])
            
            if baseline:
                target = torch.max(policy, 1).indices
            else:
                target = torch.multinomial(policy, 1).view(self.batch_dim)

            #print(target)

            route = torch.cat((route, target.unsqueeze(0)), dim = 0)
            if torch.all(target == 0):
                break

            log_prob = torch.log(policy[self.batch_arr, target]) # value of policy at each target[i]
            final_log_prob += log_prob

            length += 1-torch.nn.functional.relu(1-target)
            loss = -(final_log_prob * (-length)).mean()

        return route, loss, final_log_prob

class TPPModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TPPModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, adj_matr, obj_list, hml, baseline = False): # default is setting as False
        hvec, hbar = self.encoder(adj_matr)
        route, loss, final_log_prob = self.decoder(hvec, hbar, adj_matr, obj_list, hml, baseline)

        return route.T, loss, final_log_prob
