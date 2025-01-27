# Imports

import numpy as np
import scipy.linalg as LA
from a_algorithms import pes
from scipy.fft import dct
import scipy
import scipy.io as sio
from argparse import ArgumentParser
import a_algorithms

parser = ArgumentParser(description='Phase plots train')
parser.add_argument('--sparsity', type=int, default=10)
# parser.add_argument('--SNR', type=int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--algo', type=str, default='ISTA')


args = parser.parse_args()

numIter = 500

SNR_l = range(10,51,2); SPR_l = range(10,31)

sparsity = args.sparsity; 
device = args.device
A = sio.loadmat(f'phase_plot_sim_data_train/A')['A']
D = np.eye(100)
alpha = (LA.norm(A, 2) ** 2) * 1.001
gam_best = 0; thr_best = 0

for sparsity in SPR_l:
    for SNR in SNR_l:
        f = open(f"./phase_plot_logs_train/{args.algo}_sparsity_{sparsity}_SNR_{SNR}.txt", "w")

        import time
        start = time.time()
        
        data = sio.loadmat(f'phase_plot_sim_data_train/sparsity_{sparsity}_noise_{SNR}')
        y = data['y'] ; x = data['x'] 

        max_SNR = -100; NMSE_best = 1000; PES_best = -10

        thr_l = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        gam_l = [2,3,4,5,7,10,12,15,17,20,25,30,35,40,50,70,100,150,200,300]

        # thr_l = [0.1,0.2]; gam_l = [4,5]
        gam_best_snr = 0; gam_best_nmse = 0; gam_best_pes = 0

        print('-'*30)
        print(args.algo)

        if args.algo == 'MCP' or args.algo == 'FMCP' or args.algo == 'TF_MCP' or args.algo == 'TF_FMCP' or args.algo == 'RTF_MCP' or args.algo == 'RTF_FMCP':

            for thr in thr_l:
                for i, gamma in enumerate(gam_l):
                    if args.algo == 'MCP':
                        x_out, x_list, time_list = a_algorithms.a_MCP(y, A, D, device, numIter, thr, gamma)
                    elif args.algo == 'FMCP':
                        x_out, x_list, time_list = a_algorithms.a_FMCP(y, A, D, device, numIter, thr, gamma)
                    elif args.algo == 'TF_MCP':
                        x_out, x_list, time_list = a_algorithms.a_TF_MCP(y, A, D, device, numIter, thr, gamma)
                    elif args.algo == 'TF_FMCP':
                        x_out, x_list, time_list = a_algorithms.a_TF_FMCP(y, A, D, device, numIter, thr, gamma)
                    elif args.algo == 'RTF_MCP':
                        x_out, x_list, time_list = a_algorithms.a_RTF_MCP(y, A, D, device, numIter, thr, gamma)
                    elif args.algo == 'RTF_FMCP':
                        x_out, x_list, time_list = a_algorithms.a_RTF_FMCP(y, A, D, device, numIter, thr, gamma)

                    N_test = y.shape[1]; SNR_list = []
                    for i in range(N_test):
                        err = np.linalg.norm(x_out[:, i] - x[:, i])
                        RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
                        if np.isnan(RSNR):
                            break
                        SNR_list.append(RSNR)

                    SNR_list_ISTA = np.array(SNR_list)
                    # x_out = x_out
                    err = np.linalg.norm(x_out - x, 'fro')
                    MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
                    PES_mean, PES_std = a_algorithms.pes(x, x_out)

                    if np.mean(SNR_list_ISTA) > max_SNR:
                        max_SNR = np.mean(SNR_list_ISTA)
                        gam_best_snr = gamma; thr_best_snr = thr
                    
                    if MSE < NMSE_best:
                        gam_best_nmse = gamma; thr_best_nmse = thr
                        NMSE_best = MSE

                    if PES_mean > PES_best:
                        PES_best = PES_mean
                        gam_best_pes = gamma; thr_best_pes = thr
        else:

            for thr in thr_l:
                if args.algo == 'ISTA':
                    x_out, x_list, time_list = a_algorithms.a_ISTA(y, A, D, device, numIter, thr)
                if args.algo == 'FISTA':
                    x_out, x_list, time_list = a_algorithms.a_FISTA(y, A, D, device, numIter, thr)
                elif args.algo == 'TF_ISTA':
                    x_out, x_list, time_list = a_algorithms.a_TF_ISTA(y, A, D, device, numIter, thr)
                elif args.algo == 'TF_FISTA':
                    x_out, x_list, time_list = a_algorithms.a_TF_FISTA(y, A, D, device, numIter, thr)
                elif args.algo == 'RTF_ISTA':
                    x_out, x_list, time_list = a_algorithms.a_RTF_ISTA(y, A, D, device, numIter, thr)
                elif args.algo == 'RTF_FISTA':
                    x_out, x_list, time_list = a_algorithms.a_RTF_FISTA(y, A, D, device, numIter, thr)

                N_test = y.shape[1]; SNR_list = []
                for i in range(N_test):
                    err = np.linalg.norm(x_out[:, i] - x[:, i])
                    RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
                    if np.isnan(RSNR):
                        break
                    SNR_list.append(RSNR)

                SNR_list_ISTA = np.array(SNR_list)
                # x_out = x_out
                err = np.linalg.norm(x_out - x, 'fro')
                MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
                PES_mean, PES_std = a_algorithms.pes(x, x_out)

                if np.mean(SNR_list_ISTA) > max_SNR:
                    max_SNR = np.mean(SNR_list_ISTA)
                    thr_best_snr = thr
                
                if MSE < NMSE_best:
                    thr_best_nmse = thr
                    NMSE_best = MSE

                if PES_mean > PES_best:
                    PES_best = PES_mean
                    thr_best_pes = thr

        end = time.time()

        best_D = {'gam_snr': gam_best_snr, 'thr_snr': thr_best_snr, 'SNR': max_SNR, 'gam_nmse': gam_best_nmse, 'thr_nmse': thr_best_nmse, 'NMSE': NMSE_best, 'gam_pes': gam_best_pes, 'thr_pes': thr_best_pes, 'PES': PES_best}
        sio.savemat(f'phase_plots_vals_train/{args.algo}_sparsity_{sparsity}_SNR_{SNR}', best_D)

        print(f'SNR: {SNR} and Sparsity: {sparsity}')
        print(f'time elapsed {round(end - start, 2)} seconds' )
        print('------- BEST SNR -------')
        print(f'Testing: my {args.algo} avg SNR: {max_SNR:.2f}')
        print(f'Best thr: {thr_best_snr} and Best gamma: {gam_best_snr}')
        print('------- BEST NMSE -------')
        print(f'Testing: my {args.algo} MSE is  { NMSE_best: .2f}')
        print(f'Best thr: {thr_best_nmse} and Best gamma: {gam_best_nmse}')
        print('------- BEST PES -------')
        print(f'Testing: my {args.algo} PES is {PES_best:.4f}')
        print(f'Best thr: {thr_best_pes} and Best gamma: {gam_best_pes}')

        f.write('------- BEST SNR -------')
        f.write(f'Testing: my {args.algo} avg SNR: {max_SNR:.2f}')
        f.write(f'Best thr: {thr_best_snr} and Best gamma: {gam_best_snr}')
        f.write('------- BEST NMSE -------')
        f.write(f'Testing: my {args.algo} MSE is { NMSE_best: .2f}')
        f.write(f'Best thr: {thr_best_nmse} and Best gamma: {gam_best_nmse}')
        f.write('------- BEST PES -------')
        f.write(f'Testing: my {args.algo} PES is  {PES_best:.4f}')
        f.write(f'Best thr: {thr_best_pes} and Best gamma: {gam_best_pes}')
