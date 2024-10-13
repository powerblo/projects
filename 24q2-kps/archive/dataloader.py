import torch
from torch.utils.data import DataLoader, Dataset

class RandomGraph(Dataset):
    def __init__(self, nodes, total_size):
        init_graph = torch.randint(0,2,(total_size,nodes,nodes))
        self.adj_graph = torch.abs((init_graph - init_graph.transpose(1,2))/2)
