import torch
import numpy as np
import scipy.linalg as LA
from a_algorithms import pes
from scipy.fft import dct
import scipy
from plotting_tools import plot_signal 
import scipy.io as sio
from argparse import ArgumentParser
# Load the data 
import a_algorithms
import time

start = time.time()

parser = ArgumentParser(description='Phase plots test')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--algo', type=str, default='ISTA')
parser.add_argument('--sparsity', type=int, default=30)
parser.add_argument('--SNR', type=int, default=50)
args = parser.parse_args()

algo = args.algo
device = args.device; numIter = 5000
sparsity = args.sparsity
SNR = args.SNR
m = 70

A = sio.loadmat(f'phase_plot_sim_data_test/A')['A']; D = np.eye(100)
f = open(f"./MCP_SNR_vs_Iter/{args.algo}_sparsity_{sparsity}_SNR_{round(SNR,2)}_meas_{m}.txt", "w")

data = sio.loadmat(f'phase_plot_sim_data_test/sparsity_{sparsity}_noise_{SNR}')
y = data['y'] ; x = data['x'] 

TF_D = sio.loadmat(f'phase_plots_vals_train/{args.algo}_sparsity_{sparsity}_SNR_{SNR}')
gam_best = TF_D['gam_nmse'][0][0]; thr_best = TF_D['thr_nmse'][0][0];

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

SNR_list = np.array(SNR_list)
# x_out = x_out
err = np.linalg.norm(x_out - x, 'fro')
NMSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
PES_mean, PES_std = a_algorithms.pes(x, x_out)
PES = PES_mean
RSNR = np.mean(SNR_list) 

MSE_list = []
with torch.no_grad():
    loss= []
    D_t = torch.tensor(A, dtype=(torch.float32))
    for x_b in x_list:
        x_b = x_b.cpu().T
        err = np.linalg.norm(x_b - x, 'fro')
        MSE_list.append(-20*np.log10(np.linalg.norm(x, 'fro')/err))

f.write(f'{algo} NMSE: {NMSE:.2f}\n' )
f.write(f'{algo} RSNR: {np.mean(SNR_list) :.2f}\n' )
f.write(f'{algo} PES: {PES_mean:.4f}\n' )
f.close()

print(f'{algo} NMSE: {NMSE:.2f}' )
print(f'{algo} RSNR: {np.mean(SNR_list) :.2f}' )
print(f'{algo} PES: {PES_mean:.4f}' )

val_dict = {'MSE_list': MSE_list, 'NMSE': NMSE, 'RSNR': np.mean(SNR_list), 'PES': PES_mean}
sio.savemat(f'MCP_SNR_vs_Iter/{algo}_sparsity_{sparsity}_SNR_{SNR}_meas_{m}', val_dict)

end = time.time()
print(f'time elapsed {round(end - start, 2)} seconds' )