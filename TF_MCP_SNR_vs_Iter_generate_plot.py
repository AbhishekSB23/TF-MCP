
import numpy as np
import scipy.linalg as LA
from a_algorithms import pes
from scipy.fft import dct
import scipy
from plotting_tools import plot_signal 
import scipy.io as sio
from plotting_tools import *
import scienceplots
from argparse import ArgumentParser

parser = ArgumentParser(description='Phase plots test')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--algo', type=str, default='ISTA')
parser.add_argument('--sparsity', type=int, default=30)
parser.add_argument('--SNR', type=int, default=50)
parser.add_argument('--ylim', type=int, default=-50)
parser.add_argument('--iter', type=int, default=5000)

args = parser.parse_args()

plt.style.use(['science','ieee'])
plt.rcParams.update({"font.family": "sans","font.serif": ["cm"],"mathtext.fontset": "cm","font.size": 26}) 

ISTA_color = 'green'; TF_ISTA_color = 'blue'; RTF_ISTA_color = 'red'; 
MCP_color = 'k'; TF_MCP_color = 'm'; RTF_MCP_color = 'c'
width = 2

ylim = args.ylim
ax = plt.gca()
m = 50; n = 100
sparsity = args.sparsity; SNR = args.SNR

val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/ISTA_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_ISTA = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/TF_ISTA_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_TF_ISTA = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/RTF_ISTA_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_RTF_ISTA = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/FISTA_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_FISTA = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/TF_FISTA_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_TF_FISTA = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/RTF_FISTA_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_RTF_FISTA = val_dict['MSE_list'][0]

val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/MCP_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_MCP = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/TF_MCP_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_TF_MCP = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/RTF_MCP_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_RTF_MCP = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/FMCP_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_FMCP = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/TF_FMCP_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_TF_FMCP = val_dict['MSE_list'][0]
val_dict = sio.loadmat(f'MCP_SNR_vs_Iter/RTF_FMCP_sparsity_{sparsity}_SNR_{SNR}_meas_{m}')
MES_list_RTF_FMCP = val_dict['MSE_list'][0]

plt.figure(figsize=(8,6))
numIter = args.iter
legend_par = False

plot_signal(range(1, 1+numIter), MES_list_ISTA[:numIter], ax=ax,
    legend_label=r'ISTA',legend_show= legend_par,
    line_width=width, plot_colour=ISTA_color, line_style='-')

plot_signal(range(1, 1+numIter), MES_list_FISTA[:numIter], ax=ax,
    legend_label=r'FISTA',legend_show= legend_par,
    line_width=width, plot_colour=ISTA_color, line_style='--')

plot_signal(range(1, 1+numIter), MES_list_TF_ISTA[:numIter], ax=ax,
    legend_label=r'TF-ISTA',legend_show= legend_par,
    line_width=width, plot_colour=TF_ISTA_color, line_style='-')

plot_signal(range(1, 1+numIter), MES_list_TF_FISTA[:numIter], ax=ax,
    legend_label=r'TF-FISTA',legend_show= legend_par,
    line_width=width, plot_colour=TF_ISTA_color, line_style='--')

plot_signal(range(1, 1+numIter), MES_list_RTF_ISTA[:numIter], ax=ax,
    legend_label=r'RTF-ISTA',legend_show= legend_par,
    line_width=width, plot_colour=RTF_ISTA_color, line_style='-')

plot_signal(range(1, 1+numIter), MES_list_RTF_FISTA[:numIter], ax=ax,
    legend_label=r'RTF-FISTA',legend_show= legend_par,
    line_width=width, plot_colour=RTF_ISTA_color, line_style='--')

plot_signal(range(1, 1+numIter), MES_list_MCP[:numIter], ax=ax,
    legend_label=r'MCP',legend_show= legend_par,
    line_width=width, plot_colour=MCP_color, line_style='-')

plot_signal(range(1, 1+numIter), MES_list_FMCP[:numIter], ax=ax,
    legend_label=r'FMCP',legend_show= legend_par,
    line_width=width, plot_colour=MCP_color, line_style='--')

plot_signal(range(1, 1+numIter), MES_list_TF_MCP[:numIter], ax=ax,
    legend_label=r'TF-MCP',legend_show= legend_par,
    line_width=width, plot_colour=TF_MCP_color, line_style='-')

plot_signal(range(1, 1+numIter), MES_list_TF_FMCP[:numIter], ax=ax,
    legend_label=r'TF-FMCP',legend_show= legend_par,
    line_width=width, plot_colour=TF_MCP_color, line_style='--')

plot_signal(range(1, 1+numIter), MES_list_RTF_MCP[:numIter], ax=ax,
    legend_label=r'RTF-MCP',legend_show= legend_par,
    line_width=width, plot_colour=RTF_MCP_color, line_style='-')

plot_signal(range(1, 1+numIter), MES_list_RTF_FMCP[:numIter], ax=ax,
    legend_label=r'RTF-FMCP',
    line_width=width, plot_colour=RTF_MCP_color, line_style='--',
    xaxis_label=r'ITERATIONS', yaxis_label=r'MSE [DB]',
    n_col=2, legend_loc='upper right', legend_show= legend_par,
    xlimits=[0,1+numIter], ylimits=[ylim, 0],
    save = f'./MCP_SNR_vs_Iter/sparsity_{sparsity}_SNR_{round(SNR,2)}_m_{m}__iter_{numIter}_plot')