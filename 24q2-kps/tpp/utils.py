import torch
import os
import torch.distributed as dist
import matplotlib.pyplot as plt
import networkx as nx
import itertools

# torch utils
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def para(source, target_class):
    return {k: v for k, v in source.items() if k in target_class.__init__.__code__.co_varnames[1:target_class.__init__.__code__.co_argcount]}

# ttest
def ospttest(r, br):
    assert r.device == br.device

    diff = r - br
    mean_diff = torch.mean(diff)
    std_diff = torch.std(diff, unbiased=True)

    n = diff.numel()
    t_stat = mean_diff / (std_diff / torch.sqrt(torch.tensor(n, device=r.device)))
    def betainc(a, b, x):
        return torch.exp(torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b) + 
                         a * torch.log(x) + b * torch.log(1 - x))
    
    def student_t_cdf(t, df):
        x = df / (df + t ** 2)
        a = torch.tensor(0.5 * df, device=t.device)
        b = torch.tensor(0.5, device=t.device)
        cdf = 0.5 * betainc(a, b, x)
        return 1 - cdf if t >= 0 else cdf

    p_value = 2 * student_t_cdf(torch.abs(t_stat), n - 1)

    return p_value

# pyplot
def plot(graph):
    plt.figure(figsize=(10, 6))
    for r in graph:
        plt.plot(r, label=f'Rank {graph.index(r)}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')

def draw_nodes(distances, path):
    N = distances.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(i + 1, N):
            if distances[i, j].item() > 0:
                G.add_edge(i, j, weight=distances[i, j].item())

    pos = nx.spring_layout(G, seed=42, k=1.5)
    edges = G.edges(data=True)
    
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2)
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    path_edges = [(path[i].item(), path[i + 1].item()) for i in range(path.shape[0] - 1)]
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=5, edge_color='r')
    
    plt.savefig('nodes.jpg')

# brute force tsp
def bf_tsp(distances):
    N = distances.size(0)
    
    nodes = list(range(1, N))
    permutations = itertools.permutations(nodes)
    
    min_distance = float('inf')
    best_path = None
    
    for perm in permutations:
        path = [0] + list(perm) + [0]
        
        distance = 0
        for i in range(len(path) - 1):
            distance += distances[path[i], path[i + 1]]
        
        if distance < min_distance:
            min_distance = distance
            best_path = path
    
    return best_path

# transportation problem