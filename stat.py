import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

path = './results/'
save_path = './hist/'

inf_r = 100
int_r = 5
int_d_start = 85 # 0
int_d_inc = 5

iterN = round((inf_r-int_d_start)/int_d_inc)

single = True

for int_ind in range(iterN):
    int_d = int_ind * int_d_inc + int_d_start

    stat_0 = torch.load(path+'dist_'+str(inf_r)+'_'+str(int_r)+'_'+str(int_d)+'_rank_0.pth').to('cpu')
    stat_1 = torch.load(path+'dist_'+str(inf_r)+'_'+str(int_r)+'_'+str(int_d)+'_rank_1.pth').to('cpu')
    stat_2 = torch.load(path+'dist_'+str(inf_r)+'_'+str(int_r)+'_'+str(int_d)+'_rank_2.pth').to('cpu')
    stat_3 = torch.load(path+'dist_'+str(inf_r)+'_'+str(int_r)+'_'+str(int_d)+'_rank_3.pth').to('cpu')

    stat = stat_0 + stat_1 + stat_2 + stat_3
    stat = np.flip(stat.numpy())

    num_bins = stat.size

    bins = np.flip(np.arccos(2*(np.arange((num_bins+1))-0.5*num_bins)/num_bins))
    #  bins = np.arange(num_bins+1)
    bin_width = np.diff(bins)

    area = np.sum(stat*bin_width)

    pdf = stat/area

    # Set global variables for plotting
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 8.0
    mpl.rcParams.update({
        'xtick.major.size': 2,
        'xtick.minor.size': 1.5,
        'xtick.major.width': 0.75,
        'xtick.minor.width': 0.75,
        'xtick.labelsize': 8.0,
        'xtick.direction': 'in',
        'xtick.top': True,
        'ytick.major.size': 2,
        'ytick.minor.size': 1.5,
        'ytick.major.width': 0.75,
        'ytick.minor.width': 0.75,
        'ytick.labelsize': 8.0,
        'ytick.direction': 'in',
        'xtick.major.pad': 2,
        'xtick.minor.pad': 2,
        'ytick.major.pad': 2,
        'ytick.minor.pad': 2,
        'ytick.right': True,
        'savefig.dpi': 600,
        'savefig.transparent': True,
        'axes.linewidth': 0.75,
        'lines.linewidth': 1.0
    })
    width = 3.4
    height = width * 0.9

    fig1, ax1 = plt.subplots(figsize=(width,height))
    ax1.set_xlabel('theta')
    ax1.set_ylabel('PDF')
    ax1.set_xticks(np.arange(0, np.pi + np.pi/4, np.pi/4),[r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    # ax1.set_yscale('log')
    ax1.bar(bins[:-1], pdf, align='edge', width = bin_width)

    plt.tight_layout(pad=1, h_pad=1, w_pad=1)
    fig1.savefig(save_path+'histogram'+str(inf_r)+'_'+str(int_r)+'_'+str(int_d)+'.png')

    if single:
        break