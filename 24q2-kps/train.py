import json
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from model import *; from utils import *; from config import *
from tqdm.auto import tqdm

def run(rank, config:Config):
    # init
    print(f"Parallel on rank {rank}.")
    pbar1 = tqdm(range(epochs*steps), desc='Progress', total=epochs*steps, leave = True, position=0, colour='blue')

    adj_matr, obj_coll, hml_coll = initdata(hml_len = 4)

    paras = para(config.hp, CommonModule)
    ModelT = (Encoder(**paras, node_dim = adj_matr.shape[0], encoder_layers = config.hp['enc_layers'], device = rank),
              PathModule(**paras, node_dim = adj_matr.shape[0], clipp = config.hp['clipp'], device = rank))

    model = TPPModel(*ModelT)
    baseline_model = TPPModel(*ModelT)
    baseline_model.load_state_dict(model.state_dict(), strict = False)

    optimiser = optim.Adam(model.parameters(), lr)

    cost_graph = []
    
    adj_matr_batch = adj_matr.unsqueeze(0).to(torch.float32) 
    obj_coll = obj_coll.to(torch.float32)

    for _ in range(epochs):
        cost_t = 0
        for _ in range(steps):
            # hml = randomly sample
            hml = hml_coll.to(torch.float32)[:config.hp['batch_dim']]
            
            route, loss, log_p = model(adj_matr_batch, obj_coll, hml)
            _, baseline, _ = model(adj_matr, obj_coll, hml, baseline = True)
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            pbar1.update()

        if ospttest(cost, baseline) < 0.05:
            baseline_model.load_state_dict(model.state_dict())
        
        cost_graph.append(cost_t)

    torch.save(model.state_dict(), 'model.pth')
    cleanup()

    return cost_graph

epochs = 20 #100
steps = 1000 #2500; decent performance for nodes ~ 5 at 100
lr = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
if __name__ == '__main__':
    use_distributed_training = False

    hyperparas = {
            'market_dim': 20,
            'product_dim': 20,
            'unif_dim': 128,
            'batch_dim': 1, #fix batch processing if possible
            'mlp_dim' : 512,
            'vec_dim': 16,
            'enc_layers': 3,
            'head_dim': 8,
            'clipp' : 10
    }
    with open('hyperparas.json', 'w') as f:
        json.dump(hyperparas, f, indent=4)

    config = Config(use_distributed_training, hyperparas)
    #loss_graph = mp.spawn(run, args=(config,), nprocs=config.world_size, join=True)
    run('cpu', config)
