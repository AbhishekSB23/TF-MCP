# Imports

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from a_algorithms import pes
from scipy.fft import dct
import scipy
from plotting_tools import plot_signal 
import scienceplots
import scipy.io as sio
import a_algorithms

device = 'cuda:0'; numIter = 5000
SNR_l = range(10,51,2); SPR_l = range(10,31)
RSNR = np.zeros([21,21]); NMSE = np.zeros([21,21]); PES = np.zeros([21,21]); 

# RSNR = np.zeros([9,9]); NMSE = np.zeros([9, 9]); PES = np.zeros([9, 9])

A = sio.loadmat(f'phase_plot_sim_data/A')['A']; D = np.eye(100)

# Thresholds meant for ISTA and FISTA
thr_d = {10:0.15, 15: 0.1, 20: 0.05, 25: 0.01, 30:0.01, 35: 0.01, 40: 0.01, 45:0.005, 50: 0.002}

for snr_idx, SNR in enumerate(SNR_l):
    if SNR < 15: thr = 0.15
    elif SNR < 20: thr = 0.1
    elif SNR < 25:  thr = 0.05
    elif SNR < 45:  thr = 0.01
    else: thr = 0.005
    
    for spr_idx, sparsity in enumerate(SPR_l):
        data = sio.loadmat(f'phase_plot_sim_data/sparsity_{sparsity}_noise_{SNR}')
        y = data['y'] ; x = data['x'] 

        x_out, x_list, time_list = a_algorithms.a_FISTA(y, A, D, device, numIter, thr)

        N_test = y.shape[1]; SNR_list = []
        for i in range(N_test):
            err = np.linalg.norm(x_out[:, i] - x[:, i])
            RSNR_ = 20*np.log10(np.linalg.norm(x[:, i])/err)
            if np.isnan(RSNR_):
                break
            SNR_list.append(RSNR_)

        SNR_list_ISTA = np.array(SNR_list)
        # x_out = x_out
        err = np.linalg.norm(x_out - x, 'fro')
        NMSE[snr_idx, spr_idx] = -20*np.log10(np.linalg.norm(x, 'fro')/err)
        PES_mean, PES_std = a_algorithms.pes(x, x_out)
        PES[snr_idx, spr_idx] = PES_mean
        RSNR[snr_idx, spr_idx] = np.mean(SNR_list_ISTA) 


ISTA_D = {'RSNR': RSNR, 'NMSE': NMSE, 'PES': PES}

sio.savemat(f'phase_plot_sim_results/FISTA2', ISTA_D)

print(RSNR.min(), RSNR.max())
print(PES.min(), PES.max())
print(NMSE.min(), NMSE.max())

ISTA_D = sio.loadmat(f'phase_plot_sim_results/FISTA2')
RSNR = ISTA_D['RSNR']; PES = ISTA_D['PES']; NMSE = ISTA_D['NMSE']

import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)

RSNR_low = 0; RSNR_high = 50
PES_low = 0.5; PES_high = 1
NMSE_low = -50; NMSE_high = 0

# SNR_l = [10, 15, 20, 25, 30, 35, 40, 45, 50]
# SPR_l = [10, 12, 15, 17, 20, 22, 25, 27, 30]

plt.figure()
plt.pcolormesh(SNR_l, SPR_l, RSNR.T, edgecolors='w',cmap="plasma", vmin=RSNR_low, vmax=RSNR_high)
# plt.tick_params(left = False, right = False , labelleft = False ,
                # labelbottom = False, bottom = False)
plt.colorbar()
plt.savefig(f'phase_plot_sim_results/FISTA_RSNR2.pdf', bbox_inches='tight')

plt.figure()
plt.pcolormesh(SNR_l, SPR_l, PES.T, edgecolors='w',cmap="plasma", vmin=PES_low, vmax=PES_high)
# plt.tick_params(left = False, right = False , labelleft = False ,
                # labelbottom = False, bottom = False)
plt.colorbar()
plt.savefig(f'phase_plot_sim_results/FISTA_PES2.pdf', bbox_inches='tight')

plt.figure()
plt.pcolormesh(SNR_l, SPR_l, NMSE.T, edgecolors='w',cmap="plasma", vmin=NMSE_low, vmax=NMSE_high)
# plt.tick_params(left = False, right = False , labelleft = False ,
                # labelbottom = False, bottom = False)
plt.colorbar()
plt.savefig(f'phase_plot_sim_results/FISTA_NMSE2.pdf', bbox_inches='tight')