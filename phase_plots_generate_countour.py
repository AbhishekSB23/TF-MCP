import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
from a_algorithms import pes
from scipy.fft import dct
import scipy
import scipy.io as sio
from argparse import ArgumentParser
# Load the data 
import a_algorithms

parser = ArgumentParser(description='Phase plots Generation')
parser.add_argument('--algo', type=str, default='ISTA')
args = parser.parse_args()

RSNR = np.zeros([21,21]); NMSE = np.zeros([21,21]); PES = np.zeros([21,21]); 
SNR_l = range(10,51,2); SPR_l = range(10,31)

X, Y = np.meshgrid(SNR_l, SPR_l)

# Thresholds meant for MCP
for snr_idx, SNR in enumerate(SNR_l):
    for spr_idx, sparsity in enumerate(SPR_l):

        TF_D = sio.loadmat(f'phase_plots_vals_train/{args.algo}_sparsity_{sparsity}_SNR_{SNR}')
        RSNR[snr_idx, spr_idx] = TF_D['SNR']; PES[snr_idx, spr_idx] = TF_D['PES']; NMSE[snr_idx, spr_idx] = TF_D['NMSE']

import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)

RSNR_low = 0; RSNR_high = 40
PES_low = 0.5; PES_high = 1
NMSE_low = -40; NMSE_high = 0

colors_scheme = 'k'

plt.figure()
plt.pcolormesh(SNR_l, SPR_l, RSNR.T, cmap="plasma", vmin=RSNR_low, vmax=RSNR_high)
# plt.tick_params(left = False, right = False , labelleft = False ,
                # labelbottom = False, bottom = False)
plt.xlabel('SNR')
plt.ylabel('SPARSITY')
plt.colorbar()

# ---------------- Add the countours ----------------------
levels = np.array([15, 20, 25, 30, 35])
CS = plt.contour(X, Y, RSNR.T,  levels = levels, colors = colors_scheme)
plt.clabel(CS, inline=True, fontsize=10, colors = colors_scheme)

# --- Save the Figure ----
plt.savefig(f'phase_plot_sim_results_train/{args.algo}_RSNR_train_countour_m50.pdf', bbox_inches='tight')

#-----------------------------------
#------------ NEW FIGURE -----------
#-----------------------------------

plt.figure()
plt.pcolormesh(SNR_l, SPR_l, PES.T, cmap="plasma", vmin=PES_low, vmax=PES_high)
# plt.tick_params(left = False, right = False , labelleft = False ,
                # labelbottom = False, bottom = False)

plt.colorbar()
plt.xlabel('SNR')
plt.ylabel('SPARSITY')

# ---------------- Add the countours ----------------------
CS = plt.contour(X, Y, RSNR.T,  levels = levels, colors = colors_scheme)
plt.clabel(CS, inline=True, fontsize=10, colors = colors_scheme)

plt.savefig(f'phase_plot_sim_results_train/{args.algo}_PES_train_countour_m50.pdf', bbox_inches='tight')

#-----------------------------------
#------------ NEW FIGURE -----------
#-----------------------------------

plt.figure()
plt.pcolormesh(SNR_l, SPR_l, NMSE.T, cmap="plasma", vmin=NMSE_low, vmax=NMSE_high)
# plt.tick_params(left = False, right = False , labelleft = False ,
                # labelbottom = False, bottom = False)
plt.colorbar()
plt.xlabel('SNR')
plt.ylabel('SPARSITY')

# ---------------- Add the countours ----------------------
CS = plt.contour(X, Y, RSNR.T,  levels = levels, colors = colors_scheme)
plt.clabel(CS, inline=True, fontsize=10, colors = colors_scheme)

plt.savefig(f'phase_plot_sim_results_train/{args.algo}_NMSE_train_countour_m50.pdf', bbox_inches='tight')

# ------------------------ Training data Results -----------------------------------

ISTA_D = sio.loadmat(f'phase_plot_sim_results_test/{args.algo}')
SNR_l = range(10,51,2); SPR_l = range(10,31)
RSNR = np.zeros([21,21]); NMSE = np.zeros([21,21]); PES = np.zeros([21,21]); 

RSNR = ISTA_D['RSNR']; PES = ISTA_D['PES']; NMSE = ISTA_D['NMSE']

import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 24}

matplotlib.rc('font', **font)

#-----------------------------------
#------------ NEW FIGURE -----------
#-----------------------------------

plt.figure()
plt.pcolormesh(SNR_l, SPR_l, RSNR.T, cmap="plasma", vmin=RSNR_low, vmax=RSNR_high)
# plt.tick_params(left = False, right = False , labelleft = False ,
                # labelbottom = False, bottom = False)
plt.xlabel('SNR')
plt.ylabel('SPARSITY')
plt.colorbar()

# ---------------- Add the countours ----------------------
CS = plt.contour(X, Y, RSNR.T,  levels = levels, colors = colors_scheme)
plt.clabel(CS, inline=True, fontsize=10, colors = colors_scheme)

plt.savefig(f'phase_plot_sim_results_test/{args.algo}_RSNR_test_countour_m50.pdf', bbox_inches='tight')

#-----------------------------------
#------------ NEW FIGURE -----------
#-----------------------------------

plt.figure()
plt.pcolormesh(SNR_l, SPR_l, PES.T, cmap="plasma", vmin=PES_low, vmax=PES_high)
# plt.tick_params(left = False, right = False , labelleft = False ,
                # labelbottom = False, bottom = False)
plt.colorbar()
plt.xlabel('SNR')
plt.ylabel('SPARSITY')

# ---------------- Add the countours ----------------------
CS = plt.contour(X, Y, RSNR.T, levels = levels, colors = colors_scheme)
plt.clabel(CS, inline=True, fontsize=10, colors = colors_scheme)

plt.savefig(f'phase_plot_sim_results_test/{args.algo}_PES_test_countour_m50.pdf', bbox_inches='tight')

#-----------------------------------
#------------ NEW FIGURE -----------
#-----------------------------------

plt.figure()
plt.pcolormesh(SNR_l, SPR_l, NMSE.T, cmap="plasma", vmin=NMSE_low, vmax=NMSE_high)
# plt.tick_params(left = False, right = False , labelleft = False ,
                # labelbottom = False, bottom = False)
plt.colorbar()
plt.xlabel('SNR')
plt.ylabel('SPARSITY')

# ---------------- Add the countours ----------------------
CS = plt.contour(X, Y, RSNR.T, levels = levels, colors = colors_scheme)
plt.clabel(CS, inline=True, fontsize=10, colors = colors_scheme)

plt.savefig(f'phase_plot_sim_results_test/{args.algo}_NMSE_test_countour_m50.pdf', bbox_inches='tight')