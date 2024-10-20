import torch
import json

from model import *; from utils import *; from config import *; from train import *

rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval(input):
    paras = para(config.hp, CommonModule)

    Common = CommonModule(**paras, device = rank)
    ModelT = (EmbeddingModule(**paras, eps = config.hp['eps'], device = rank),
              MarketEncoder(**paras, enc_layers = config.hp['enc_layers'], device = rank),
              PathModule(**paras, clipp = config.hp['clipp'], device = rank))

    model = TPPModel(*ModelT)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    with torch.no_grad():
        route, _, _ = model(*input)

    return route

def random_cost(md, bd):
    return torch.randint(low = 0, high = 10, size=(bd,md,md), device = rank).to(dtype = torch.float32)

def no_transport(md, pd, bd):
    supply = torch.zeros((pd, md), device = rank).unsqueeze(0).repeat(bd, 1, 1)
    demand = torch.zeros(pd, device = rank).unsqueeze(0).repeat(bd, 1)
    price = torch.zeros(md, device = rank).unsqueeze(0).repeat(bd, 1, 1)

    cost_raw = random_cost(md, bd)
    cost = (cost_raw + cost_raw.transpose(-2,-1))/2

    return supply, price, demand, cost

if __name__ == '__main__':
    use_distributed_training = True 
    with open('hyperparas.json', 'r') as f:
        hyperparas = json.load(f)

    config = Config(use_distributed_training, hyperparas)

    s, d, p, c = no_transport(hyperparas['market_dim'],
        hyperparas['product_dim'],
        hyperparas['batch_dim'])
    
    paths = eval((s, d, p, c))

    for i in range(10):
        path_md = paths[i]
        path_tr = bf_tsp(c[i])

        print('path_md : ', path_md)
        print('path_tr : ', path_tr)

    #draw_nodes(c[0], path_md, path_tr)
