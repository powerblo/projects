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
        self.head_dim = head_dim
        self.head_num = head_num

        self.device = device
        
        self.batch_arr = torch.arange(self.batch_dim, device = self.device)

    def compat(self, Wq, Wk, Wv, vec):
        qv, kv, vv = ( # vectors : batch dim x node dim x head num x head dim
        torch.einsum('kij,abk->abij', Wq, vec),
        torch.einsum('kij,abk->abij', Wk, vec),
        torch.einsum('kij,abk->abij', Wv, vec))
        
        compat = torch.einsum('aibk,ajbk->aijb',qv,kv)/np.sqrt(self.head_dim) # compat : batch dim x node dim x node dim x head num

        return compat, vv

    def attention(self, compat, vv, Wo):
        weight = F.softmax(compat, dim = 2) # weight : batch dim x node dim x node dim x head num
        received_vec = torch.einsum('aijb,ajkb->aikb', weight, vv) # recvec : batch dim x node dim x head num x head dim
        mha = torch.einsum('ijk,alij->alk', Wo, received_vec) # mha : batch dim x node dim x unif dim
        return mha 

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

        self.bn1 = nn.ModuleList([nn.BatchNorm1d(self.market_dim, device = self.device) for _ in range(self.enc_layers)])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(self.market_dim, device = self.device) for _ in range(self.enc_layers)])

    def forward(self, adj_matr): # adj matrix : batch dim x node dim x node dim
        hvec_l = self.init_embed(adj_matr) # embedding : batch dim x node dim x unif dim
        for l in range(self.enc_layers):
            compat, vv_l = self.compat(self.qv_p[l], self.kv_p[l], self.vv_p[l], hvec_l)
            compat_mask = compat[:,adj_matr] # mask non adjacent nodes 
            mha_l = self.attention(compat_mask, vv_l, self.ov_p[l])

            hhat_l = self.bn1[l](hvec_l + mha_l)
            hvec_l = self.bn2[l](hhat_l + self.mha_mlp[l](hhat_l))

        hbar_l = torch.sum(hvec_l, dim = 1)/(1+self.market_dim) # hbar : batch dim x unif dim

        return hvec_l, hbar_l

class PathModule(CommonModule):
    def __init__(self, clipp, **kwargs):
        super(PathModule, self).__init__(**kwargs)
        self.clipp = clipp

        # Parameters : Decoder LSTM + MHA
        self.qv_p = nn.Parameter(torch.rand((self.unif_dim, self.head_num, self.head_dim), device = self.device)) 
        self.kv_p = nn.Parameter(torch.rand((self.unif_dim, self.head_num, self.head_dim), device = self.device)) 
        self.vv_p = nn.Parameter(torch.rand((self.unif_dim, self.head_num, self.head_dim), device = self.device)) 
        self.ov_p = nn.Parameter(torch.rand((self.head_num, self.head_dim, self.unif_dim), device = self.device)) 
        
        self.vec_1, self.vec_f = (nn.Parameter(torch.rand(self.unif_dim)), nn.Parameter(torch.rand(self.unif_dim)))

    def MaskAtt(self, compat, route, objectives):
        exclude_l = route.T.clone().detach()

        if torch.any(exclude_l > 0):
            for i in range(1,exclude_l.shape[1]):
                compat[self.batch_arr,exclude_l[:,i]] = -float('inf')
        
        for i in range(objectives.shape[0]):
            if not (route == objectives[i]).any():
                compat[:,0] = -float('inf')
                break
        # if all u_dm are -inf; all places are visited, stay at 0; zero contr. to cost
        route[torch.all(compat == -float('inf'), dim = 1), 0] = 1
        return route

    def forward(self, hvec, hbar, baseline = False):
        route = torch.zeros(1, self.batch_dim , dtype = torch.int, device = self.device)
        # run decoder
        for t in range(self.node_dim):
            if route.shape[0] == 1:
                vec_concat = torch.concat((hbar, self.vec_1, self.vec_f), dim = 1)
            else:
                vec_concat = torch.concat((hbar, hvec[route[t-1]], hvec[route[1]]))

            qv, kv, vv = ( 
            torch.einsum('kij,ak->aij', self.qv_p, vec_concat), # batch dim x head num x head dim
            torch.einsum('kij,abk->abij', self.kv_p, hvec), # batch dim x node dim x head num x head dim
            torch.einsum('kij,abk->abij', self.vv_p, hvec))

            compat = self.clipp * torch.tanh(torch.einsum('aij,abij->aib',qv,kv)/np.sqrt(self.head_dim)) # batch dim x head num x node dim
            compat_mask = self.MaskAtt(compat)
            weight = F.softmax(compat_mask, dim = 1) # weight : batch dim x head num x node dim
            # (c) x node dim  X   
            received_vec = torch.einsum('aib,ajkb->aikb', weight, vv) # recvec : batch dim x head num x head dim
            mha = torch.einsum('ijk,alij->alk', Wo, received_vec) # mha : batch dim x node dim x unif dim

            
            
            if baseline:
                a_t = torch.max(policy, 1).indices
            else:
                a_t = torch.multinomial(policy, 1).view(self.batch_dim)
            pi_t = torch.cat((pi_t, a_t.unsqueeze(0)), dim = 0)
            if torch.all(a_t == 0):
                break
        
            log_prob = torch.log(policy[self.batch_arr, a_t])
            final_log_prob += log_prob

        return pi_t, final_log_prob

class TPPModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TPPModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, adj_matr, objectives, baseline = False):
        hvec, hbar = self.encoder(adj_matr)
        pi_t, final_log_prob = self.decoder(hvec, hbar, baseline)


        return pi_t.T, final_log_prob
