import json
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from model import *; from utils import *; from config import *

from tqdm.auto import tqdm

def run(rank, config:Config):
    # init
    print(f"Parallel on rank {rank}.")
    setup(rank, config.world_size)
    pbar1 = tqdm(range(epochs*steps), desc='Progress', total=epochs*steps, leave = True, position=0, colour='blue')
    #scaler = torch.cuda.amp.GradScaler()
    
    # generate module objects, parameter passing
    ModelT = (Encoder(**paras, enc_layers = config.hp['enc_layers'], device = rank),
              PathModule(**paras, clipp = config.hp['clipp'], device = rank))

    model = TPPModel(*ModelT)
    baseline_model = TPPModel(*ModelT)
    baseline_model.load_state_dict(model.state_dict(), strict = False)

    optimiser = optim.Adam(model.parameters(), lr)

    cost_graph = []

    for _ in range(epochs):
        cost_t = 0
        for _ in range(steps):
            route, cost, log_p = model(supply_tr, price_tr, demand_tr, cost_tr)
            _, baseline, _ = baseline_model(supply_tr, price_tr, demand_tr, cost_tr, baseline = True)

            loss = torch.mean((cost - baseline) * log_p)
            cost_t += torch.mean(cost).item()/steps
            
            optimiser.zero_grad()
            #scaler.scale(loss).backward()
            #scaler.step(optimiser)
            #scaler.update()
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
    use_distributed_training = True

    hyperparas = {
            'market_dim': 20,
            'product_dim': 20,
            'unif_dim': 128,
            'batch_dim': 64, #512
            'mlp_dim' : 512,
            'vec_dim': 16,
            'enc_layers': 3,
            'head_dim': 8,
            'clipp' : 10
    }
    with open('hyperparas.json', 'w') as f:
        json.dump(hyperparas, f, indent=4)

    config = Config(use_distributed_training, hyperparas)
    loss_graph = mp.spawn(run, args=(config,), nprocs=config.world_size, join=True)
    plot(loss_graph)
