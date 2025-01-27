#%% Import the libraries

import numpy as np
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from plotting_tools import *

#%% Load the convergence plots for sparsity 20 noise 1e-3 

# parser = ArgumentParser(description='Plot-SNR-vs-Iterations')
# parser.add_argument('--sparsity', type=int, default=20, help='% of non-zeros in the sparse vector')
# parser.add_argument('--noise_level', type=float, default=0.01, help='Noise Level')
# parser.add_argument('--device', type=str, default='cuda:0', help='The GPU id')
# parser.add_argument('--sensing', type=str, default='normal', help='Sensing matrix type')
# args = parser.parse_args()


# step = 0.2
sparsity = 20 #args.sparsity
noise_level = 0.001 # args.noise_level
sensing = 'normal' #args.sensing

thr_dict = {0.001: 0.01, 0.005: 0.01, 0.003: 0.01, 0.01: 0.01, 0.03: 0.03, 0.05: 0.05, 0.1:0.1, 0:0.001}
iter_dict = {0.001: 10000, 0.005: 5000, 0.003: 5000, 0.01: 5000, 0.03: 3000, 0.05: 1000, 0.1:1000, 0:100000}

thr_ = thr_dict[noise_level]
numIter = iter_dict[noise_level]
# RSNR_dict['RSNR'] = SNR_mean_list

# ------------------------------------------------------------------------------------------
# Load the convergence plots for sparsity 20 noise 1e-3 

import scipy.io as sio
PTF_RSNR_dict = sio.loadmat(f'stored_results/PTF_ISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
UNTF_RSNR_dict = sio.loadmat(f'stored_results/UNTF_ISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RSNR_dict = sio.loadmat(f'stored_results/ISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
MCP_dict = sio.loadmat(f'stored_results/MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
TF_MCP_dict = sio.loadmat(f'stored_results/TF_MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RTF_MCP_dict = sio.loadmat(f'stored_results/RTF_MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
FMCP_dict = sio.loadmat(f'stored_results/FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
TF_FMCP_dict = sio.loadmat(f'stored_results/TF_FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RTF_FMCP_dict = sio.loadmat(f'stored_results/RTF_FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

F_PTF_RSNR_dict = sio.loadmat(f'stored_results/PTF_FISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
F_UNTF_RSNR_dict = sio.loadmat(f'stored_results/UNTF_FISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
F_RSNR_dict = sio.loadmat(f'stored_results/FISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

# AMP_dict = sio.loadmat(f'stored_results/AMP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

PTF_RSNR = PTF_RSNR_dict['RSNR'].reshape(-1,)
UNTF_RSNR = UNTF_RSNR_dict['RSNR'].reshape(-1,)
RSNR = RSNR_dict['RSNR'].reshape(-1,)
MCP = MCP_dict['RSNR'].reshape(-1,)
TF_MCP = TF_MCP_dict['RSNR'].reshape(-1,)
RTF_MCP = RTF_MCP_dict['RSNR'].reshape(-1,)
FMCP = FMCP_dict['RSNR'].reshape(-1,)
TF_FMCP = TF_FMCP_dict['RSNR'].reshape(-1,)
RTF_FMCP = RTF_FMCP_dict['RSNR'].reshape(-1,)

F_PTF_RSNR = F_PTF_RSNR_dict['RSNR'].reshape(-1,)
F_UNTF_RSNR = F_UNTF_RSNR_dict['RSNR'].reshape(-1,)
F_RSNR = F_RSNR_dict['RSNR'].reshape(-1,)

cut_off = 1e-6

conv = 700
for i in range(1, len(MCP)):
    if abs(MCP[i] - MCP[i-1]) < cut_off:
        MCP = MCP[:i+200]
        break

for i in range(1, len(TF_MCP)):
    if abs(TF_MCP[i] - TF_MCP[i-1]) < cut_off:
        TF_MCP = TF_MCP[:i+conv]
        break

for i in range(1, len(RTF_MCP)):
    if abs(RTF_MCP[i] - RTF_MCP[i-1]) < cut_off:
        RTF_MCP = RTF_MCP[:i+conv]
        break

for i in range(1, len(FMCP)):
    if abs(FMCP[i] - FMCP[i-1]) < cut_off:
        FMCP = FMCP[:i+conv]
        break

for i in range(1, len(TF_FMCP)):
    if abs(TF_FMCP[i] - TF_FMCP[i-1]) < cut_off:
        TF_FMCP = TF_FMCP[:i+conv]
        break

for i in range(1, len(RTF_FMCP)):
    if abs(RTF_FMCP[i] - RTF_FMCP[i-1]) < cut_off:
        RTF_FMCP = RTF_FMCP[:i+conv]
        break
#----------------------------------------------------------
# Plot the convergence plots

plt.style.use(['science','ieee'])
plt.rcParams.update({"font.family": "sans","font.serif": ["cm"],"mathtext.fontset": "cm","font.size": 22}) 

plt.figure(figsize=(8,6))
ax = plt.gca()

xmax = 3100 # max(len(PTF_RSNR), len(RSNR), len(UNTF_RSNR)) + 1000
legend = False; anotate = False

x1 = np.arange(len(RSNR))
plot_signal(x1, -RSNR, ax=ax,
    legend_label=r'ISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RSNR[-1] -2])

x1 = np.arange(len(MCP))
plot_signal(x1, MCP, ax=ax,
    legend_label=r'MCP', legend_show= legend,
    line_width=2, plot_colour='m', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -MCP[-1] -2])

x1 = np.arange(len(TF_MCP))
plot_signal(x1, TF_MCP, ax=ax,
    legend_label=r'TF-MCP', legend_show= legend,
    line_width=2, plot_colour='k', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -TF_MCP[-1] -2])

x1 = np.arange(len(RTF_MCP))
plot_signal(x1, RTF_MCP, ax=ax,
    legend_label=r'RTF-MCP', legend_show= legend,
    line_width=2, plot_colour='c', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RTF_MCP[-1] -2])

x1 = np.arange(len(TF_FMCP))
plot_signal(x1, TF_FMCP, ax=ax,
    legend_label=r'TF-FMCP', legend_show= legend,
    line_width=2, plot_colour='k', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -TF_MCP[-1] -2])

x1 = np.arange(len(RTF_FMCP))
plot_signal(x1, RTF_FMCP, ax=ax,
    legend_label=r'RTF-FMCP', legend_show= legend,
    line_width=2, plot_colour='c', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RTF_MCP[-1] -2])

x2 = np.arange(len(PTF_RSNR))
plot_signal(x2, -PTF_RSNR, ax=ax,
    legend_label=r'TF-ISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='-', annotates = anotate, 
    annotation = r'$6.48$ s ', pos = [x2[-1], -PTF_RSNR[-1]])

x3 = np.arange(len(UNTF_RSNR))
plot_signal(x3, -UNTF_RSNR, ax=ax,
    legend_label=r'RTF-ISTA', legend_show= legend,
    line_width=2, plot_colour='red', line_style='-', annotates = anotate, 
    annotation = r'$10.11$ s ', pos = [x2[-1], -UNTF_RSNR[-1]])

x1 = np.arange(len(F_RSNR))
plot_signal(x1, -F_RSNR, ax=ax,
    legend_label=r'FISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='--', annotates = anotate, 
    annotation = r'$15.87$ s ', pos = [x1[-1], -F_RSNR[-1] + 2])

x2 = np.arange(len(F_PTF_RSNR))
plot_signal(x2, -F_PTF_RSNR, ax=ax,
    legend_label=r'TF-FISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='--', annotates = anotate, 
    annotation = r'$3.41$ s ', pos = [x2[-1], -F_PTF_RSNR[-1]])

x3 = np.arange(len(F_UNTF_RSNR))
plot_signal(x3, -F_UNTF_RSNR, ax=ax,
    xaxis_label=r'ITERATIONS', yaxis_label= r'NMSE [dB]',
    legend_label=r'RTF-FISTA', legend_show= legend,
    n_col=2, legend_loc='upper right',
    line_width=2, plot_colour='red', line_style='--',
    xlimits=[0,xmax], ylimits=[-55, 0] , annotates = anotate, 
    annotation = r'$3.51$ s ', pos = [x3[-1], -F_UNTF_RSNR[-1]], save = f'figures/FMCP_noise_{noise_level}_sparsity_{sparsity}_sensing_{sensing}_{xmax}')

#%% New plots for sparsity 25 noise 0.001


# step = 0.2
sparsity = 25 #args.sparsity
noise_level = 0.001 # args.noise_level
sensing = 'normal' #args.sensing

thr_dict = {0.001: 0.01, 0.005: 0.01, 0.003: 0.01, 0.01: 0.01, 0.03: 0.03, 0.05: 0.05, 0.1:0.1, 0:0.001}
iter_dict = {0.001: 10000, 0.005: 5000, 0.003: 5000, 0.01: 5000, 0.03: 3000, 0.05: 1000, 0.1:1000, 0:100000}

thr_ = thr_dict[noise_level]
numIter = iter_dict[noise_level]
# RSNR_dict['RSNR'] = SNR_mean_list

# ------------------------------------------------------------------------------------------
# Load the convergence plots for sparsity 20 noise 1e-3 
import scipy.io as sio
PTF_RSNR_dict = sio.loadmat(f'stored_results/PTF_ISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
UNTF_RSNR_dict = sio.loadmat(f'stored_results/UNTF_ISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RSNR_dict = sio.loadmat(f'stored_results/ISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
MCP_dict = sio.loadmat(f'stored_results/MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
TF_MCP_dict = sio.loadmat(f'stored_results/TF_MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RTF_MCP_dict = sio.loadmat(f'stored_results/RTF_MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
FMCP_dict = sio.loadmat(f'stored_results/FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
TF_FMCP_dict = sio.loadmat(f'stored_results/TF_FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RTF_FMCP_dict = sio.loadmat(f'stored_results/RTF_FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

F_PTF_RSNR_dict = sio.loadmat(f'stored_results/PTF_FISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
F_UNTF_RSNR_dict = sio.loadmat(f'stored_results/UNTF_FISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
F_RSNR_dict = sio.loadmat(f'stored_results/FISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

# AMP_dict = sio.loadmat(f'stored_results/AMP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

PTF_RSNR = PTF_RSNR_dict['RSNR'].reshape(-1,)
UNTF_RSNR = UNTF_RSNR_dict['RSNR'].reshape(-1,)
RSNR = RSNR_dict['RSNR'].reshape(-1,)
MCP = MCP_dict['RSNR'].reshape(-1,)
TF_MCP = TF_MCP_dict['RSNR'].reshape(-1,)
RTF_MCP = RTF_MCP_dict['RSNR'].reshape(-1,)
FMCP = FMCP_dict['RSNR'].reshape(-1,)
TF_FMCP = TF_FMCP_dict['RSNR'].reshape(-1,)
RTF_FMCP = RTF_FMCP_dict['RSNR'].reshape(-1,)

F_PTF_RSNR = F_PTF_RSNR_dict['RSNR'].reshape(-1,)
F_UNTF_RSNR = F_UNTF_RSNR_dict['RSNR'].reshape(-1,)
F_RSNR = F_RSNR_dict['RSNR'].reshape(-1,)

cut_off = 1e-6

conv = 700
for i in range(1, len(MCP)):
    if abs(MCP[i] - MCP[i-1]) < cut_off:
        MCP = MCP[:i+200]
        break

for i in range(1, len(TF_MCP)):
    if abs(TF_MCP[i] - TF_MCP[i-1]) < cut_off:
        TF_MCP = TF_MCP[:i+conv]
        break

for i in range(1, len(RTF_MCP)):
    if abs(RTF_MCP[i] - RTF_MCP[i-1]) < cut_off:
        RTF_MCP = RTF_MCP[:i+conv]
        break

for i in range(1, len(FMCP)):
    if abs(FMCP[i] - FMCP[i-1]) < cut_off:
        FMCP = FMCP[:i+conv]
        break

for i in range(1, len(TF_FMCP)):
    if abs(TF_FMCP[i] - TF_FMCP[i-1]) < cut_off:
        TF_FMCP = TF_FMCP[:i+conv]
        break

for i in range(1, len(RTF_FMCP)):
    if abs(RTF_FMCP[i] - RTF_FMCP[i-1]) < cut_off:
        RTF_FMCP = RTF_FMCP[:i+conv]
        break

#----------------------------------------------------------
# Plot the convergence plots

plt.style.use(['science','ieee'])
plt.rcParams.update({"font.family": "sans","font.serif": ["cm"],"mathtext.fontset": "cm","font.size": 22}) 

plt.figure(figsize=(8,6))
ax = plt.gca()

xmax = 5100 # max(len(PTF_RSNR), len(RSNR), len(UNTF_RSNR)) + 1000
legend = False; anotate = False

x1 = np.arange(len(RSNR))
plot_signal(x1, -RSNR, ax=ax,
    legend_label=r'ISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RSNR[-1] -2])

x1 = np.arange(len(MCP))
plot_signal(x1, MCP, ax=ax,
    legend_label=r'MCP', legend_show= legend,
    line_width=2, plot_colour='m', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -MCP[-1] -2])

x1 = np.arange(len(TF_MCP))
plot_signal(x1, TF_MCP, ax=ax,
    legend_label=r'TF-MCP', legend_show= legend,
    line_width=2, plot_colour='k', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -TF_MCP[-1] -2])

x1 = np.arange(len(RTF_MCP))
plot_signal(x1, RTF_MCP, ax=ax,
    legend_label=r'RTF-MCP', legend_show= legend,
    line_width=2, plot_colour='c', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RTF_MCP[-1] -2])

x1 = np.arange(len(TF_FMCP))
plot_signal(x1, TF_FMCP, ax=ax,
    legend_label=r'TF-FMCP', legend_show= legend,
    line_width=2, plot_colour='k', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -TF_MCP[-1] -2])

x1 = np.arange(len(RTF_FMCP))
plot_signal(x1, RTF_FMCP, ax=ax,
    legend_label=r'RTF-FMCP', legend_show= legend,
    line_width=2, plot_colour='c', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RTF_MCP[-1] -2])

x2 = np.arange(len(PTF_RSNR))
plot_signal(x2, -PTF_RSNR, ax=ax,
    legend_label=r'TF-ISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='-', annotates = anotate, 
    annotation = r'$6.48$ s ', pos = [x2[-1], -PTF_RSNR[-1]])

x3 = np.arange(len(UNTF_RSNR))
plot_signal(x3, -UNTF_RSNR, ax=ax,
    legend_label=r'RTF-ISTA', legend_show= legend,
    line_width=2, plot_colour='red', line_style='-', annotates = anotate, 
    annotation = r'$10.11$ s ', pos = [x2[-1], -UNTF_RSNR[-1]])

x1 = np.arange(len(F_RSNR))
plot_signal(x1, -F_RSNR, ax=ax,
    legend_label=r'FISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='--', annotates = anotate, 
    annotation = r'$15.87$ s ', pos = [x1[-1], -F_RSNR[-1] + 2])

x2 = np.arange(len(F_PTF_RSNR))
plot_signal(x2, -F_PTF_RSNR, ax=ax,
    legend_label=r'TF-FISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='--', annotates = anotate, 
    annotation = r'$3.41$ s ', pos = [x2[-1], -F_PTF_RSNR[-1]])

x3 = np.arange(len(F_UNTF_RSNR))
plot_signal(x3, -F_UNTF_RSNR, ax=ax,
    xaxis_label=r'ITERATIONS', yaxis_label= r'NMSE [dB]',
    legend_label=r'RTF-FISTA', legend_show= legend,
    n_col=2, legend_loc='upper right',
    line_width=2, plot_colour='red', line_style='--',
    xlimits=[0,xmax], ylimits=[-55, 0] , annotates = anotate, 
    annotation = r'$3.51$ s ', pos = [x3[-1], -F_UNTF_RSNR[-1]], save = f'figures/FMCP_noise_{noise_level}_sparsity_{sparsity}_sensing_{sensing}_{xmax}')


#%% Sparisty 30 Noise 0.001



# step = 0.2
sparsity = 30 #args.sparsity
noise_level = 0.001 # args.noise_level
sensing = 'normal' #args.sensing

thr_dict = {0.001: 0.01, 0.005: 0.01, 0.003: 0.01, 0.01: 0.01, 0.03: 0.03, 0.05: 0.05, 0.1:0.1, 0:0.001}
iter_dict = {0.001: 10000, 0.005: 5000, 0.003: 5000, 0.01: 5000, 0.03: 3000, 0.05: 1000, 0.1:1000, 0:100000}

thr_ = thr_dict[noise_level]
numIter = iter_dict[noise_level]
# RSNR_dict['RSNR'] = SNR_mean_list

# ------------------------------------------------------------------------------------------
# Load the convergence plots for sparsity 20 noise 1e-3 
import scipy.io as sio
PTF_RSNR_dict = sio.loadmat(f'stored_results/PTF_ISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
UNTF_RSNR_dict = sio.loadmat(f'stored_results/UNTF_ISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RSNR_dict = sio.loadmat(f'stored_results/ISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
MCP_dict = sio.loadmat(f'stored_results/MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
TF_MCP_dict = sio.loadmat(f'stored_results/TF_MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RTF_MCP_dict = sio.loadmat(f'stored_results/RTF_MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
FMCP_dict = sio.loadmat(f'stored_results/FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
TF_FMCP_dict = sio.loadmat(f'stored_results/TF_FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
RTF_FMCP_dict = sio.loadmat(f'stored_results/RTF_FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

F_PTF_RSNR_dict = sio.loadmat(f'stored_results/PTF_FISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
F_UNTF_RSNR_dict = sio.loadmat(f'stored_results/UNTF_FISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')
F_RSNR_dict = sio.loadmat(f'stored_results/FISTA_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

# AMP_dict = sio.loadmat(f'stored_results/AMP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations')

PTF_RSNR = PTF_RSNR_dict['RSNR'].reshape(-1,)
UNTF_RSNR = UNTF_RSNR_dict['RSNR'].reshape(-1,)
RSNR = RSNR_dict['RSNR'].reshape(-1,)
MCP = MCP_dict['RSNR'].reshape(-1,)
TF_MCP = TF_MCP_dict['RSNR'].reshape(-1,)
RTF_MCP = RTF_MCP_dict['RSNR'].reshape(-1,)
FMCP = FMCP_dict['RSNR'].reshape(-1,)
TF_FMCP = TF_FMCP_dict['RSNR'].reshape(-1,)
RTF_FMCP = RTF_FMCP_dict['RSNR'].reshape(-1,)

F_PTF_RSNR = F_PTF_RSNR_dict['RSNR'].reshape(-1,)
F_UNTF_RSNR = F_UNTF_RSNR_dict['RSNR'].reshape(-1,)
F_RSNR = F_RSNR_dict['RSNR'].reshape(-1,)

cut_off = 1e-6

conv = 700
for i in range(1, len(MCP)):
    if abs(MCP[i] - MCP[i-1]) < cut_off:
        MCP = MCP[:i+200]
        break

for i in range(1, len(TF_MCP)):
    if abs(TF_MCP[i] - TF_MCP[i-1]) < cut_off:
        TF_MCP = TF_MCP[:i+conv]
        break

for i in range(1, len(RTF_MCP)):
    if abs(RTF_MCP[i] - RTF_MCP[i-1]) < cut_off:
        RTF_MCP = RTF_MCP[:i+conv]
        break

for i in range(1, len(FMCP)):
    if abs(FMCP[i] - FMCP[i-1]) < cut_off:
        FMCP = FMCP[:i+conv]
        break

for i in range(1, len(TF_FMCP)):
    if abs(TF_FMCP[i] - TF_FMCP[i-1]) < cut_off:
        TF_FMCP = TF_FMCP[:i+conv]
        break

for i in range(1, len(RTF_FMCP)):
    if abs(RTF_FMCP[i] - RTF_FMCP[i-1]) < cut_off:
        RTF_FMCP = RTF_FMCP[:i+conv]
        break

#----------------------------------------------------------
# Plot the convergence plots

plt.style.use(['science','ieee'])
plt.rcParams.update({"font.family": "sans","font.serif": ["cm"],"mathtext.fontset": "cm","font.size": 22}) 

plt.figure(figsize=(8,6))
ax = plt.gca()

xmax = 9100 # max(len(PTF_RSNR), len(RSNR), len(UNTF_RSNR)) + 1000
legend = False; anotate = False

x1 = np.arange(len(RSNR))
plot_signal(x1, -RSNR, ax=ax,
    legend_label=r'ISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RSNR[-1] -2])

x1 = np.arange(len(MCP))
plot_signal(x1, MCP, ax=ax,
    legend_label=r'MCP', legend_show= legend,
    line_width=2, plot_colour='m', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -MCP[-1] -2])

x1 = np.arange(len(FMCP))
plot_signal(x1, FMCP, ax=ax,
    legend_label=r'FMCP', legend_show= legend,
    line_width=2, plot_colour='m', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -MCP[-1] -2])

x1 = np.arange(len(TF_MCP))
plot_signal(x1, TF_MCP, ax=ax,
    legend_label=r'TF-MCP', legend_show= legend,
    line_width=2, plot_colour='k', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -TF_MCP[-1] -2])

x1 = np.arange(len(RTF_MCP))
plot_signal(x1, RTF_MCP, ax=ax,
    legend_label=r'RTF-MCP', legend_show= legend,
    line_width=2, plot_colour='c', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RTF_MCP[-1] -2])

x1 = np.arange(len(TF_FMCP))
plot_signal(x1, TF_FMCP, ax=ax,
    legend_label=r'TF-FMCP', legend_show= legend,
    line_width=2, plot_colour='k', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -TF_MCP[-1] -2])

x1 = np.arange(len(RTF_FMCP))
plot_signal(x1, RTF_FMCP, ax=ax,
    legend_label=r'RTF-FMCP', legend_show= legend,
    line_width=2, plot_colour='c', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RTF_MCP[-1] -2])

x2 = np.arange(len(PTF_RSNR))
plot_signal(x2, -PTF_RSNR, ax=ax,
    legend_label=r'TF-ISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='-', annotates = anotate, 
    annotation = r'$6.48$ s ', pos = [x2[-1], -PTF_RSNR[-1]])

x3 = np.arange(len(UNTF_RSNR))
plot_signal(x3, -UNTF_RSNR, ax=ax,
    legend_label=r'RTF-ISTA', legend_show= legend,
    line_width=2, plot_colour='red', line_style='-', annotates = anotate, 
    annotation = r'$10.11$ s ', pos = [x2[-1], -UNTF_RSNR[-1]])

x1 = np.arange(len(F_RSNR))
plot_signal(x1, -F_RSNR, ax=ax,
    legend_label=r'FISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='--', annotates = anotate, 
    annotation = r'$15.87$ s ', pos = [x1[-1], -F_RSNR[-1] + 2])

x2 = np.arange(len(F_PTF_RSNR))
plot_signal(x2, -F_PTF_RSNR, ax=ax,
    legend_label=r'TF-FISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='--', annotates = anotate, 
    annotation = r'$3.41$ s ', pos = [x2[-1], -F_PTF_RSNR[-1]])

x3 = np.arange(len(F_UNTF_RSNR))
plot_signal(x3, -F_UNTF_RSNR, ax=ax,
    xaxis_label=r'ITERATIONS', yaxis_label= r'NMSE [dB]',
    legend_label=r'RTF-FISTA', legend_show= legend,
    n_col=2, legend_loc='upper right',
    line_width=2, plot_colour='red', line_style='--',
    xlimits=[0,xmax], ylimits=[-55, 0] , annotates = anotate, 
    annotation = r'$3.51$ s ', pos = [x3[-1], -F_UNTF_RSNR[-1]], save = f'figures/FMCP_noise_{noise_level}_sparsity_{sparsity}_sensing_{sensing}_{xmax}')

#%% Plot the Legend

#----------------------------------------------------------
# Plot the convergence plots

plt.style.use(['science','ieee'])
plt.rcParams.update({"font.family": "sans","font.serif": ["cm"],"mathtext.fontset": "cm","font.size": 25}) 

plt.figure(figsize=(8,6))
ax = plt.gca()

xmax = 3100 # max(len(PTF_RSNR), len(RSNR), len(UNTF_RSNR)) + 1000
legend = True; anotate = False

x1 = np.arange(len(RSNR))
plot_signal(x1, -RSNR, ax=ax,
    legend_label=r'ISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RSNR[-1] -2])

x1 = np.arange(len(MCP))
plot_signal(x1, MCP, ax=ax,
    legend_label=r'MCP', legend_show= legend,
    line_width=2, plot_colour='m', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -MCP[-1] -2])

x1 = np.arange(len(FMCP))
plot_signal(x1, FMCP, ax=ax,
    legend_label=r'FMCP', legend_show= legend,
    line_width=2, plot_colour='m', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -MCP[-1] -2])

x1 = np.arange(len(TF_MCP))
plot_signal(x1, TF_MCP, ax=ax,
    legend_label=r'TF-MCP', legend_show= legend,
    line_width=2, plot_colour='k', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -TF_MCP[-1] -2])

x1 = np.arange(len(RTF_MCP))
plot_signal(x1, RTF_MCP, ax=ax,
    legend_label=r'RTF-MCP', legend_show= legend,
    line_width=2, plot_colour='c', line_style='-', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RTF_MCP[-1] -2])

x1 = np.arange(len(TF_FMCP))
plot_signal(x1, TF_FMCP, ax=ax,
    legend_label=r'TF-FMCP', legend_show= legend,
    line_width=2, plot_colour='k', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -TF_MCP[-1] -2])

x1 = np.arange(len(RTF_FMCP))
plot_signal(x1, RTF_FMCP, ax=ax,
    legend_label=r'RTF-FMCP', legend_show= legend,
    line_width=2, plot_colour='c', line_style='--', annotates = anotate, 
    annotation = r'$14.07$ s ', pos = [x1[-1], -RTF_MCP[-1] -2])

x2 = np.arange(len(PTF_RSNR))
plot_signal(x2, -PTF_RSNR, ax=ax,
    legend_label=r'TF-ISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='-', annotates = anotate, 
    annotation = r'$6.48$ s ', pos = [x2[-1], -PTF_RSNR[-1]])

x3 = np.arange(len(UNTF_RSNR))
plot_signal(x3, -UNTF_RSNR, ax=ax,
    legend_label=r'RTF-ISTA', legend_show= legend,
    line_width=2, plot_colour='red', line_style='-', annotates = anotate, 
    annotation = r'$10.11$ s ', pos = [x2[-1], -UNTF_RSNR[-1]])

x1 = np.arange(len(F_RSNR))
plot_signal(x1, -F_RSNR, ax=ax,
    legend_label=r'FISTA', legend_show= legend,
    line_width=2, plot_colour='green', line_style='--', annotates = anotate, 
    annotation = r'$15.87$ s ', pos = [x1[-1], -F_RSNR[-1] + 2])

x2 = np.arange(len(F_PTF_RSNR))
plot_signal(x2, -F_PTF_RSNR, ax=ax,
    legend_label=r'TF-FISTA', legend_show= legend,
    line_width=2, plot_colour='blue', line_style='--', annotates = anotate, 
    annotation = r'$3.41$ s ', pos = [x2[-1], -F_PTF_RSNR[-1]])

x3 = np.arange(len(F_UNTF_RSNR))
plot_signal(x3, -F_UNTF_RSNR, ax=ax,
    legend_label=r'RTF-FISTA', legend_show= legend,
    n_col=2, legend_loc='upper right',
    line_width=2, plot_colour='red', line_style='--',
    xlimits=[0,xmax], ylimits=[0, 50] , annotates = anotate, 
    annotation = r'$3.51$ s ', pos = [x3[-1], -F_UNTF_RSNR[-1]], save = f'figures/FMCP_legend')
