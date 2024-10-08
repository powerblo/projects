import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from scipy.constants import pi

# path = './flpass/results/' # Use this for first passage
path = './flpass/results_alt/' # Use this for last passage

# Set global variables for plotting
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

pos = 0 # Initial conducting sphere position
rad = 5 # Conducting sphere radius
rad_inf = 100 # Infinity sphere radius, R
incr = 0.25 # Increment of moving conducting sphere; d = pos + i * incr; d < rad
incr_count = 20 # pos + incr_count < rad & rad_inf - rad

def thr(i, theta):
    d = pos + incr * i
    if d != 0:
        norm = d / (rad + d - np.abs(rad - d))
    else:
        norm = 0.5
    return norm * 1 / np.sqrt(rad**2 + d**2 + 2 * d * rad * np.cos(theta)) * 10

def ocfp(i, theta):
    R = rad_inf - (pos + incr * i)
    alpha = rad / (rad + R)
    return (1 - alpha**2)/(4*pi*(1-2*alpha*np.cos(theta)+alpha**2)**(3/2))

range1 = range(incr_count)
pbar1 = tqdm(range1, desc='Plotting in Progress', total=incr_count, leave = True, position=0, colour='blue')

for i in range1:
    stat_0 = torch.load(path+'distribution_rank_0'+'_'+str(i)+'.pth').to('cpu')
    stat_1 = torch.load(path+'distribution_rank_1'+'_'+str(i)+'.pth').to('cpu')
    stat_2 = torch.load(path+'distribution_rank_2'+'_'+str(i)+'.pth').to('cpu')
    stat_3 = torch.load(path+'distribution_rank_3'+'_'+str(i)+'.pth').to('cpu')

    stat = stat_0 + stat_1 + stat_2 + stat_3
    stat = stat.numpy() # stat[0] = number of points between z = [-1, -1+2/num_bin], stat[-1] = number of points between z = [1-2/num_bin, 1]

    num_bins = stat.size

    bins = np.arccos(2*(np.arange((num_bins+1))-0.5*num_bins)/num_bins)
    # bins = np.arange(num_bins+1)
    bin_width = np.diff(bins)

    # print(np.sum(stat))

    pdf = stat / np.sum(-stat*bin_width)

    theory = thr(i, bins[:-1])
    theory = theory / np.sum(-theory*bin_width)

    fig1, ax1 = plt.subplots(figsize=(width,height))
    ax1.set_xlabel('theta')
    ax1.set_ylabel('Density')
    # ax1.set_yscale('log')
    ax1.set_xticks(np.arange(0, np.pi + np.pi/4, np.pi/4),[r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax1.bar(bins[:-1], pdf, align='edge', width = bin_width)
    ax1.plot(bins[:-1], theory, color = 'red', linewidth = 0.5)

    plt.tight_layout(pad=1, h_pad=1, w_pad=1)
    # fig1.savefig('./flpass/plots_z/histogram_'+'_'+str(i)+'.png') # Use this for first passage
    fig1.savefig('./flpass/plots_z_last/histogram_'+'_'+str(i)+'.png') # Use this for last passage
    pbar1.update(1)