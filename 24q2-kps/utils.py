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

