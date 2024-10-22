import torch
import os
import torch.distributed as dist
import pickle

# torch utils
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

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

def initdata(hml_len):
    with open("adj_mat.data", "rb") as r:
        adj_mat = pickle.load(r)
    adj_mat_ts = torch.tensor(adj_mat)

    with open("df_frames.data", "rb") as r:
        frames_data = pickle.load(r)
    fir = frames_data.drop(columns=['index']).map(lambda x: int(x[1]))
    sec = frames_data.drop(columns=['index']).map(lambda x: int(x[4]))
    vals = fir*8 + sec
    obj_ts = torch.tensor(vals.values)

    #with open(f"~/workspace/rawdata/framedata/h_db_{hml_len}.data", "rb") as r:
    with open(f"C:\\Users\\hanse\\Documents\\h_db_{hml_len}.data", "rb") as r:
        hml_data = pickle.load(r)
    hml_data_l = [item['sym'] for item in hml_data]
    vals = [[8*x+y for x, y in item] for item in hml_data_l]
    hml_coll_ts = torch.tensor(vals)

    return adj_mat_ts, obj_ts, hml_coll_ts 
