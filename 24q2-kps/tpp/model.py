import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CommonModule(nn.Module):
    def __init__(self, market_dim, product_dim, unif_dim, mlp_dim, vec_dim, head_dim, batch_dim, device):
        super(CommonModule, self).__init__()
        # external model specs
        self.market_dim = market_dim
        self.product_dim = product_dim
        # internal model specs
        self.unif_dim = unif_dim
        self.mlp_dim = mlp_dim
        self.batch_dim = batch_dim
        # mha specs
        self.vec_dim = vec_dim
        self.head_dim = head_dim

        self.device = device
        
        self.market_arr = torch.arange(self.market_dim, device = self.device)
        self.batch_arr = torch.arange(self.batch_dim, device = self.device)

    def MHAVec(self, layer, input):
        return layer(input).view((self.batch_dim, input.shape[1], self.vec_dim, self.head_dim))
    
    def MHA(self, qv, kv, vv, ov):
        u = torch.einsum('aibk,ajbk->aijk',qv,kv)/np.sqrt(self.vec_dim)
        att = torch.einsum('aibk,abjk->aijk',F.softmax(u, dim = 2),vv)
        mha = ov(att.reshape((self.batch_dim, att.shape[1], self.vec_dim*self.head_dim)))
        return mha

    def GenerateFeatures(self, max_supply, max_price, para_l):
        m_nodes = torch.randint(0, 1000, (self.batch_dim, self.market_dim, 2), device = self.device) # random x y

        s_rnd = torch.randint(1, self.market_dim + 1, (self.batch_dim, self.product_dim,), device = self.device)
        s_ind = torch.rand(self.batch_dim, self.market_dim, self.product_dim, device = self.device).argsort(dim=1).transpose(-2,-1)
        s_msk = self.market_arr.expand(self.batch_dim, self.product_dim, self.market_dim) < s_rnd.unsqueeze(2)

        s_feat = torch.zeros(self.batch_dim, self.product_dim, self.market_dim, dtype = torch.float32, device = self.device)
        s_feat.scatter_(2, s_ind, s_msk.float()) # supply : unlimited
        # if supply is zero for a product at all markets for all products; extremely unlikely!

        s_feat[:, :, 0] = 0.0 # depot

        # market feature
        c_feat = torch.sqrt(((m_nodes.unsqueeze(2) - m_nodes.unsqueeze(1))**2).sum(dim=3)) # node distance ~ travelling cost; necessarily euclidean?

        # edge features
        s_feat = s_feat * (torch.rand(self.batch_dim, self.product_dim, self.market_dim, device = self.device)*(max_supply-1) + 1) # supply : limited
        p_feat = torch.rand(self.batch_dim, self.product_dim, self.market_dim, device = self.device)*(max_price-1) + 1 # price
        p_feat[:, :, 0] = 0.0 # depot

        # product features
        s_max, _ = torch.max(s_feat, dim = 2)
        d_feat = para_l * s_max + (1-para_l) * torch.sum(s_feat, dim = 2) # demand

        return s_feat, p_feat, d_feat, c_feat

class EmbeddingModule(CommonModule):
    def __init__(self, **kwargs):
        super(EmbeddingModule, self).__init__(**kwargs)
        self.eps = 1e-6

        # Parameters
        self.mlp_m = nn.Sequential(
                nn.Linear(self.market_dim, self.mlp_dim, device = self.device),
                nn.ReLU(),
                nn.Linear(self.mlp_dim, self.market_dim, device = self.device)
                )
        self.mlp_p = nn.Sequential(
                nn.Linear(self.product_dim, self.mlp_dim, device = self.device),
                nn.ReLU(),
                nn.Linear(self.mlp_dim, self.product_dim, device = self.device)
                )

    def forward(self, s_feat, p_feat, d_feat, c_feat):
        #s_feat = F.normalize(s_feat, p=2, dim=2)
        #p_feat = F.normalize(p_feat, p=2, dim=2)
        #d_feat = F.normalize(d_feat, p=2, dim=1)

        # inital embeddings
        m_init = nn.Linear(self.market_dim, self.unif_dim, device = self.device)(c_feat)
        e_init = nn.Linear(2, self.unif_dim, device = self.device)(torch.stack((s_feat, p_feat), dim = 3))
        p_init = nn.Linear(1, self.unif_dim, device = self.device)(d_feat.unsqueeze(1).transpose(-2,-1))

        p_agg = torch.sum(F.relu(m_init.unsqueeze(1) + e_init), dim = 2)
        p_upd = self.mlp_p(((1 + self.eps) * p_init + p_agg).transpose(-2,-1)).transpose(-2,-1)
        
        m_agg = torch.sum(F.relu(p_init.unsqueeze(2) + e_init), dim = 1)
        m_upd = self.mlp_m(((1 + self.eps) * m_init + m_agg).transpose(-2,-1)).transpose(-2,-1)

        return p_upd, m_upd

class MarketEncoder(CommonModule):
    def __init__(self, enc_layers, **kwargs):
        super(MarketEncoder, self).__init__(**kwargs)
        self.enc_layers = enc_layers

        # Paramters
        self.mlp_u = nn.Sequential(
                nn.Linear(self.unif_dim, self.mlp_dim, device = self.device),
                nn.ReLU(),
                nn.Linear(self.mlp_dim, self.unif_dim, device = self.device)
                )

        ## Market encoder MHA
        self.qv_p = nn.ModuleList([nn.Linear(self.unif_dim, self.vec_dim*self.head_dim, device = self.device) for _ in range(self.enc_layers)])
        self.kv_p = nn.ModuleList([nn.Linear(self.unif_dim, self.vec_dim*self.head_dim, device = self.device) for _ in range(self.enc_layers)])
        self.vv_p = nn.ModuleList([nn.Linear(self.unif_dim, self.vec_dim*self.head_dim, device = self.device) for _ in range(self.enc_layers)])
        self.ov_p = nn.ModuleList([nn.Linear(self.vec_dim*self.head_dim, self.unif_dim, device = self.device) for _ in range(self.enc_layers)])

        self.bn1 = nn.ModuleList([nn.BatchNorm1d(self.market_dim, device = self.device) for _ in range(self.enc_layers)])
        self.bn2 = nn.ModuleList([nn.BatchNorm1d(self.market_dim, device = self.device) for _ in range(self.enc_layers)])

    def forward(self, m_upd):
        m_l = m_upd
        for l in range(self.enc_layers):
            # check if view and reshape work as expected
            qv_l, kv_l, vv_l = (
            self.MHAVec(self.qv_p[l], m_l),
            self.MHAVec(self.kv_p[l], m_l), 
            self.MHAVec(self.vv_p[l], m_l))

            mha_l = self.MHA(qv_l, kv_l, vv_l, self.ov_p[l])
            mhat_l = self.bn1[l](m_l + mha_l)
            m_l = self.bn2[l](mhat_l + self.mlp_u(mhat_l))

        m_gemb = torch.sum(m_l, dim = 1)/(1+self.market_dim)

        return m_l, m_gemb

class PathModule(CommonModule):
    def __init__(self, clipp, **kwargs):
        super(PathModule, self).__init__(**kwargs)
        self.clipp = clipp

        # Parameters : Decoder LSTM + MHA
        self.lstm = nn.LSTM(self.market_dim, self.market_dim, device = self.device)
        
        self.qv_d = nn.Linear(1, self.vec_dim*self.head_dim, device = self.device)
        self.kv_d = nn.Linear(1, self.vec_dim*self.head_dim, device = self.device)
        self.vv_d = nn.Linear(1, self.vec_dim*self.head_dim, device = self.device)
        self.ov_d = nn.Linear(self.vec_dim*self.head_dim, 1, device = self.device)

    def Decoder(self, mcont, mN):
        mcont, mN = mcont.unsqueeze(1).transpose(-2,-1), mN.unsqueeze(1).transpose(-2,-1)
        
        qv, kv, vv = (
        self.MHAVec(self.qv_d, mcont),
        self.MHAVec(self.kv_d, mN), 
        self.MHAVec(self.vv_d, mN))
        mcont_p = self.MHA(qv, kv, vv, self.ov_d)
        qv_p = self.MHAVec(self.qv_d, mcont_p)
        u_dm = self.clipp * torch.tanh(torch.einsum('aibk,ajbk->aijk',qv_p,kv)/np.sqrt(self.vec_dim))
        return u_dm
    
    def RemainingDemand(self, demand, supply, route):
        rd = demand
        for i in route:
            rd = rd - supply.transpose(-2,-1)[self.batch_arr.to(dtype=torch.long),i.to(dtype=torch.long)]
        return rd
    
    def RouteContext(self, a_t, o_t, c_t, m_N):
        o_t, (_, c_t) = self.lstm(
            m_N.transpose(-2,-1)[self.batch_arr, a_t].unsqueeze(1).permute(1,0,2).contiguous(), 
            (o_t, c_t))
        return o_t, c_t

    def MaskScore(self, u_dm, pi_t, rd):
        exclude_l = pi_t.T.clone().detach()
        if torch.any(exclude_l > 0):
            for i in range(1,exclude_l.shape[1]):
                u_dm[self.batch_arr,exclude_l[:,i]] = -float('inf')
        if torch.any(rd > 0):
            u_dm[:,0] = -float('inf')
        # if all u_dm are -inf; all places are visited, stay at 0; zero contr. to cost
        u_dm[torch.all(u_dm == -float('inf'), dim = 1), 0] = 1
        return u_dm

    def forward(self, s, d, m_gemb, m_N, p_upd, baseline):
        # initialise route context
        pi_t = torch.zeros(1, self.batch_dim , dtype = torch.int, device = self.device)
        o_t = torch.zeros(1, self.batch_dim, self.market_dim, device = self.device)
        c_t = torch.zeros(1, self.batch_dim, self.market_dim, device = self.device)
        a_t = 0
        final_log_prob = 0
        
        # run decoder
        while True:
            rd = self.RemainingDemand(d, s, pi_t)

            p_t = torch.einsum('ai,aik->ak',rd, p_upd)
            o_t, c_t = self.RouteContext(a_t, o_t, c_t, m_N)
            m_context = torch.cat((m_gemb, p_t, o_t[0]), dim = 1)
            
            u_dm = torch.sum(self.Decoder(m_context, m_N.transpose(-2,-1)[self.batch_arr,a_t]), dim = (1,3))
            u_dm = self.MaskScore(u_dm, pi_t, rd)
            policy = F.softmax(u_dm, dim = 1)
            
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
    def __init__(self, embedder, encoder, pathmodule):
        super(TPPModel, self).__init__()
        
        self.embedder = embedder
        self.encoder = encoder
        self.pathmodule = pathmodule

    def TravelCost(self, c, pi_t):
        batch_format = self.pathmodule.batch_arr.expand(pi_t.shape[0],self.pathmodule.batch_dim).to(dtype=torch.long)
        
        path_format = pi_t[:, :].to(dtype=torch.long)
        path_next_format = torch.cat((pi_t[1:,:],torch.zeros(1,self.pathmodule.batch_dim, device=self.pathmodule.device)),dim=0).type(torch.long)
        consec_cost = c[batch_format, path_format, path_next_format]
        
        return consec_cost.sum(dim=0)

    def forward(self, s, p, d, c, baseline = False):
        # run initial embedding
        p_upd, m_upd = self.embedder(s, p, d, c)
        m_N, m_gemb = self.encoder(m_upd)
        pi_t, final_log_prob = self.pathmodule(s, d, m_gemb, m_N, p_upd, baseline)

        cost = self.TravelCost(c, pi_t)

        return pi_t.T, cost, final_log_prob
