import numpy as np
import scipy.linalg as LA
from a_algorithms import pes
from scipy.fft import dct
import scipy
import scipy.io as sio
from argparse import ArgumentParser
# Load the data 
import a_algorithms
import time
import os
start = time.time()

parser = ArgumentParser(description='Phase plots test')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--algo', type=str, default='ISTA')
args = parser.parse_args()


device = args.device; numIter = 5000
SNR_l = range(10,51,2); SPR_l = range(10,31)
RSNR = np.zeros([21,21]); NMSE = np.zeros([21,21]); PES = np.zeros([21,21]); 

# RSNR = np.zeros([9,9]); NMSE = np.zeros([9, 9]); PES = np.zeros([9, 9])

A = sio.loadmat(f'phase_plot_sim_data_test/A')['A']; D = np.eye(100)
os.makedirs("phase_plot_sim_results_test",exist_ok=True)
# Thresholds meant for MCP
for snr_idx, SNR in enumerate(SNR_l):    
    for spr_idx, sparsity in enumerate(SPR_l):
        data = sio.loadmat(f'phase_plot_sim_data_test/sparsity_{sparsity}_noise_{SNR}')
        y = data['y'] ; x = data['x'] 

        TF_D = sio.loadmat(f'phase_plots_vals_train/{args.algo}_sparsity_{sparsity}_SNR_{SNR}')
        thr_best = TF_D['thr_snr'][0][0]; gam_best = TF_D['gam_snr'][0][0]

        if args.algo == 'ISTA':
            x_out, x_list, time_list = a_algorithms.a_ISTA(y, A, D, device, numIter, thr_best)
        elif args.algo == 'FISTA':
            x_out, x_list, time_list = a_algorithms.a_FISTA(y, A, D, device, numIter, thr_best)
        elif args.algo == 'TF_ISTA':
            x_out, x_list, time_list = a_algorithms.a_TF_ISTA(y, A, D, device, numIter, thr_best)
        elif args.algo == 'TF_FISTA':
            x_out, x_list, time_list = a_algorithms.a_TF_FISTA(y, A, D, device, numIter, thr_best)
        elif args.algo == 'RTF_ISTA':
            x_out, x_list, time_list = a_algorithms.a_RTF_ISTA(y, A, D, device, numIter, thr_best)
        elif args.algo == 'RTF_FISTA':
            x_out, x_list, time_list = a_algorithms.a_RTF_FISTA(y, A, D, device, numIter, thr_best)
        elif args.algo == 'MCP':
            x_out, x_list, time_list = a_algorithms.a_MCP(y, A, D, device, numIter, thr_best, gam_best)
        elif args.algo == 'FMCP':
            x_out, x_list, time_list = a_algorithms.a_FMCP(y, A, D, device, numIter, thr_best, gam_best)
        elif args.algo == 'TF_MCP':
            x_out, x_list, time_list = a_algorithms.a_TF_MCP(y, A, D, device, numIter, thr_best, gam_best)
        elif args.algo == 'TF_FMCP':
            x_out, x_list, time_list = a_algorithms.a_TF_FMCP(y, A, D, device, numIter, thr_best, gam_best)
        elif args.algo == 'RTF_MCP':
            x_out, x_list, time_list = a_algorithms.a_RTF_MCP(y, A, D, device, numIter, thr_best, gam_best)
        elif args.algo == 'RTF_FMCP':
            x_out, x_list, time_list = a_algorithms.a_RTF_FMCP(y, A, D, device, numIter, thr_best, gam_best)
        else:
            print(args.algo)
            print("-"*10, "Wrong Algorithm Choosen", "-"*10)
            exit()

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

sio.savemat(f'phase_plot_sim_results_test/{args.algo}', {'RSNR': RSNR, 'NMSE': NMSE, 'PES': PES})

end = time.time()
print(f'time elapsed {round(end - start, 2)} seconds' )