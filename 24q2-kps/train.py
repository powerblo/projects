import json
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from model import *; from utils import *; from config import *
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, random_split

def run(rank, config:Config, hml_len, epochs):
    # init
    lr = 1e-2
    pbar1 = tqdm(range(epochs), desc='Progress', total=epochs, leave = True, position=0, colour='blue')

    adj_matr, obj_coll, hml_coll = initdata(hml_len)
    bd = config.hp['batch_dim']
    #adj_matr, obj_coll, hml_coll = adj_matr.to(rank), obj_coll.to(rank), hml_coll.to(rank)

    paras = para(config.hp, CommonModule)
    ModelT = (Encoder(**paras, node_dim = adj_matr.shape[0], encoder_layers = config.hp['enc_layers'], device = rank),
            PathModule(**paras, node_dim = adj_matr.shape[0], clipp = config.hp['clipp'], device = rank))

    model = TPPModel(*ModelT)
    optimiser = optim.Adam(model.parameters(), lr)

    train_size, eval_size, test_size = int(hml_coll.shape[0]*0.4), int(hml_coll.shape[0]*0.4), int(hml_coll.shape[0]*0.4)
    train_size -= train_size % bd
    eval_size -= eval_size % bd
    test_size -= test_size % bd

    hml_rand = hml_coll[torch.randperm(hml_coll.shape[0], device=rank)[:(train_size+eval_size+test_size)]]
    hml_train, hml_eval, hml_test = hml_rand[:train_size],hml_rand[train_size:train_size+eval_size],hml_rand[train_size+eval_size:]

    #print(train_size, eval_size, test_size)
    adj_matr_batch = adj_matr.unsqueeze(0).to(torch.float32)

    loss_graph = []
    length_graph = []

    for _ in range(epochs):
        total_loss = 0
        for i in range(int(hml_train.shape[0]/bd)):
            route, loss, log_p = model(adj_matr_batch, obj_coll, hml_train[i*bd:(i+1)*bd])
            total_loss += loss
            
        optimiser.zero_grad()
        total_loss.backward()
        optimiser.step()

        eval_total_loss = 0
        eval_total_length = 0
        for i in range(int(hml_eval.shape[0]/bd)):
            eval_route, eval_loss, _ = model(adj_matr_batch, obj_coll, hml_eval[i*bd:(i+1)*bd])
            eval_total_loss += eval_loss.item()
            eval_total_length += eval_route.shape[1]*torch.mean(torch.min(torch.ones_like(eval_route),eval_route.to(torch.float32))).item()

        loss_graph.append(eval_total_loss)
        length_graph.append(eval_total_length)
        pbar1.update()
    
    torch.save(model.state_dict(), f'model_{hml_len}.pth')

    route_test_total = []
    for i in range(int(hml_test.shape[0]/bd)):
        route_test, _, _ = model(adj_matr_batch, obj_coll, hml_test[i*bd:(i+1)*bd])
        route_test_total.append(route_test)
    route_test_total = torch.stack(route_test_total)

    torch.save(hml_test, f'hml_test_{hml_len}.pth')
    torch.save(route_test_total, f'route_test_{hml_len}.pth')

    loss_graph = torch.tensor(loss_graph)
    length_graph = torch.tensor(length_graph)
    torch.save(loss_graph, f'loss_graph_{hml_len}.pth')
    torch.save(length_graph, f'length_graph_{hml_len}.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    use_distributed_training = False

    torch.set_default_device('cpu')

    hyperparas = {
            'market_dim': 20,
            'product_dim': 20,
            'unif_dim': 128,
            'batch_dim': 256, #fix batch processing if possible
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
    for hml_len in range(7,8):
        run('cpu', config, hml_len, epochs = 2)
