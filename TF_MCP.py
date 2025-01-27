
#%% Imports

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

#%% Data generation

sparsity = 30; device = 'cuda:0'
numIter = 3000; SNR = 55; y_lim = -55
noise_level = 0.001

iter_dict = {0.001: 10000, 0.005: 5000, 0.003: 5000, 0.01: 5000, 0.03: 3000, 0.05: 1000, 0.1:1000, 0:100000}
numIter = iter_dict[noise_level]


multi = 10
thr_ISTA = 0.01 * multi
thr_TF = 0.001 * 70

thr_MCP = 0.01
gam_MCP = 500

thr_FMCP = 0.01
gam_FMCP = 500

thr_TF_MCP = 0.003
gam_TF_MCP = 500

thr_TF_FMCP = 0.003
gam_TF_FMCP = 500

thr_RTF_MCP = 0.002
gam_RTF_MCP = 500

thr_RTF_FMCP = 0.002
gam_RTF_FMCP = 500


m = 70; n = 100; Ex = 100

# D = np.random.randn(n, n)                       # random gaussian dictionary
# D, _ = LA.qr(D)                                 # Get an orthogonal matrix
# D = LA.sqrtm(LA.inv(D @ D.T)) @ D

# D = dct(np.eye(n), axis = 0, norm='ortho')  # DCT dictionary
D = np.eye(n)

seed = 45; rng = np.random.RandomState(seed)
al = rng.normal(0, 1, (n, Ex)) * rng.binomial(1, sparsity/100, (n, Ex))

for i in range(Ex):
    if LA.norm(al[:, i]) < 1e-2 :
        print('0 norm sparse code')

A = rng.normal(0, 1/np.sqrt(m), [m, n])
A /= LA.norm(A, 2, axis = 0)
x = D.T @ al
y1 = A @ x 
noise = rng.normal(0, 1, y1.shape)

# noise_level = LA.norm(y1, 2) / ((10 ** (SNR / 20)) * LA.norm(noise, 2))

y = y1 + noise_level * noise

print(noise_level)

SNR_ = 20 * np.log10(LA.norm(y1, 2)/ LA.norm(noise_level * noise, 2))
print(SNR_)
f = open(f"./TF_MCP/sparsity_{sparsity}_SNR_{round(SNR,2)}_meas_{m}.txt", "w")
f.write(f'Sparsity = {sparsity/100} ; Noise = {round(noise_level, 4)}; Noise SNR = {round(SNR, 2)}\nMeasurements = {m} ; signal length = {n}\nThreshold ISTA = {thr_ISTA} ; Threshold TF ISTA = {thr_TF} \nThreshold MCP = {thr_MCP} ; Gamma MCP = {gam_MCP} \nThreshold TF MCP = {thr_TF_MCP} ; Gamma TF MCP = {gam_TF_MCP} \n')

# ----------------------------------------------------------------------------------------

# %% analysis ISTA 

import a_algorithms
import importlib
importlib.reload(a_algorithms)

# device = 'cuda:0'
# import time
# start = time.time()
# x_out, x_list, time_list = a_algorithms.a_ISTA(y, A, D, device, numIter, thr_ISTA)
# end = time.time()
# print(f'time elapsed {round(end - start, 2)} seconds' )

# N_test = y.shape[1]; SNR_list = []
# for i in range(N_test):
#     err = np.linalg.norm(x_out[:, i] - x[:, i])
#     RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
#     if np.isnan(RSNR):
#         break
#     SNR_list.append(RSNR)

# SNR_list_ISTA = np.array(SNR_list)
# # x_out = x_out
# err = np.linalg.norm(x_out - x, 'fro')
# MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
# PES_mean, PES_std = a_algorithms.pes(x, x_out)

# print('Testing: my ISTA MSE is ', round(MSE, 2))
# print(f'Testing: my ISTA avg SNR: { np.mean(SNR_list_ISTA):.2f} std: {np.std(SNR_list_ISTA):.2f}')
# print(f'Testing: my ISTA avg PES: { PES_mean:.4f} std: {PES_std:.4f}')

# f.write( '-'* 50 + '\n')
# f.write(f'ISTA MSE = {round(MSE, 2)} dB \nISTA SNR mean = {round(np.mean(SNR_list_ISTA), 2)} dB \nISTA SNR std = {round(np.std(SNR_list_ISTA), 2)} dB \nISTA avg PES: { PES_mean:.4f} std: {PES_std:.4f}\nTime = {round(end - start, 2)} seconds\n')

# SNR_mean_list_ISTA = []; RSNR_dict = {}

# with torch.no_grad():
#     loss= []
#     D_t = torch.tensor(A, dtype=(torch.float32))
#     for x_b in x_list:
#         x_b = x_b.cpu().T
#         err = np.linalg.norm(x_b - x, 'fro')
#         RSNR = -20*np.log10(np.linalg.norm(x, 'fro')/err)
#         SNR_mean_list_ISTA.append(RSNR)

# # ----------------------------------------------------------------------------------------
# # %% analysis TF - ISTA 

# import importlib
# importlib.reload(a_algorithms)

# device = 'cuda:1'
# import time
# start = time.time()
# x_out, x_list, time_list = a_algorithms.a_TF_ISTA(y, A, D, device, numIter, thr_TF)
# end = time.time()
# print(f'time elapsed {round(end - start, 2)} seconds' )

# N_test = y.shape[1]; SNR_list = []
# for i in range(N_test):
#     err = np.linalg.norm(x_out[:, i] - x[:, i])
#     RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
#     if np.isnan(RSNR):
#         break
#     SNR_list.append(RSNR)

# SNR_list_TF_ISTA = np.array(SNR_list)

# # x_out = x_out.cpu().T
# err = np.linalg.norm(x_out - x, 'fro')
# MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
# PES_mean, PES_std = a_algorithms.pes(x, x_out)

# print('Testing: my TF ISTA MSE is ', round(MSE, 2))
# print(f'Testing: my TF ISTA avg SNR: { np.mean(SNR_list_TF_ISTA):.2f} std: {np.std(SNR_list_TF_ISTA):.2f}')
# print(f'Testing: my TF ISTA avg PES: { PES_mean:.4f} std: {PES_std:.4f}')

# f.write( '-'* 50 + '\n')
# f.write(f'TF ISTA MSE = {round(MSE, 2)} dB \nTF ISTA SNR mean = {round(np.mean(SNR_list_TF_ISTA), 2)} dB \nTF ISTA SNR std = {round(np.std(SNR_list_TF_ISTA), 2)} dB \nTF ISTA avg PES: { PES_mean:.4f} std: {PES_std:.4f}\nTime = {round(end - start, 2)} seconds\n')

# SNR_mean_list_TF_ISTA = []; RSNR_dict = {}

# with torch.no_grad():
#     loss= []
#     D_t = torch.tensor(A, dtype=(torch.float32))
#     for x_b in x_list:
#         x_b = x_b.cpu().T
#         err = np.linalg.norm(x_b - x, 'fro')
#         RSNR = -20*np.log10(np.linalg.norm(x, 'fro')/err)
#         SNR_mean_list_TF_ISTA.append(RSNR)

# ----------------------------------------------------------------------------------------

# %% analysis MCP

import importlib
importlib.reload(a_algorithms)

device = 'cuda:0'
import time
start = time.time()
x_out, x_list, time_list = a_algorithms.a_MCP(y, A, D, device, numIter, thr_MCP, gam_MCP)
end = time.time()
print(f'time elapsed {round(end - start, 2)} seconds' )

N_test = y.shape[1]; SNR_list = []
for i in range(N_test):
    err = np.linalg.norm(x_out[:, i] - x[:, i])
    RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
    if np.isnan(RSNR):
        break
    SNR_list.append(RSNR)

SNR_list_MCP = np.array(SNR_list)

# x_out = x_out.cpu().T
err = np.linalg.norm(x_out - x, 'fro')
MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
PES_mean, PES_std = a_algorithms.pes(x, x_out)

print('Testing: my MCP MSE is ', round(MSE, 2))
print(f'Testing: my MCP avg SNR: { np.mean(SNR_list_MCP):.2f} std: {np.std(SNR_list_MCP):.2f}')
print(f'Testing: my MCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}')

f.write( '-'* 50 + '\n')
f.write(f'MCP MSE = {round(MSE, 2)} dB \nMCP SNR mean = {round(np.mean(SNR_list_MCP), 2)} dB \nMCP SNR std = {round(np.std(SNR_list_MCP), 2)} dB \nMCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}\nTime = {round(end - start, 2)} seconds\n')


SNR_mean_list_MCP = []; RSNR_dict = {}

with torch.no_grad():
    loss= []
    D_t = torch.tensor(A, dtype=(torch.float32))
    for x_b in x_list:
        x_b = x_b.cpu().T
        err = np.linalg.norm(x_b - x, 'fro')
        RSNR = -20*np.log10(np.linalg.norm(x, 'fro')/err)
        SNR_mean_list_MCP.append(RSNR)

# MCP_dict = {'RSNR': SNR_mean_list_MCP}
# import scipy.io as sio
# sensing = 'normal'
# sio.savemat(f'stored_results/MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations', MCP_dict)

# ----------------------------------------------------------------------------------------

# %% analysis TF MCP

import importlib
importlib.reload(a_algorithms)


device = 'cuda:1'
import time
start = time.time()
x_out, x_list, time_list = a_algorithms.a_TF_MCP(y, A, D, device, numIter, thr_TF_MCP, gam_TF_MCP)
end = time.time()
print(f'time elapsed {round(end - start, 2)} seconds' )

N_test = y.shape[1]; SNR_list = []
for i in range(N_test):
    err = np.linalg.norm(x_out[:, i] - x[:, i])
    RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
    if np.isnan(RSNR):
        break
    SNR_list.append(RSNR)

SNR_list_TF_MCP = np.array(SNR_list)

# x_out = x_out.cpu().T
err = np.linalg.norm(x_out - x, 'fro')
MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
PES_mean, PES_std = a_algorithms.pes(x, x_out)

print('Testing: my TF MCP MSE is ', round(MSE, 2))
print(f'Testing: my TF MCP avg SNR: { np.mean(SNR_list_TF_MCP):.2f} std: {np.std(SNR_list_TF_MCP):.2f}')
print(f'Testing: my TF MCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}')
f.write( '-'* 50 + '\n')
f.write(f'TF MCP MSE = {round(MSE, 2)} dB \nTF MCP avg SNR: { np.mean(SNR_list_TF_MCP):.2f} std: {np.std(SNR_list_TF_MCP):.2f}\nTF MCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}\nTime = {round(end - start, 2)} seconds\n')


SNR_mean_list_TF_MCP = []; RSNR_dict = {}

with torch.no_grad():
    loss= []
    D_t = torch.tensor(A, dtype=(torch.float32))
    for x_b in x_list:
        x_b = x_b.cpu().T
        err = np.linalg.norm(x_b - x, 'fro')
        RSNR = -20*np.log10(np.linalg.norm(x, 'fro')/err)
        SNR_mean_list_TF_MCP.append(RSNR)

# TF_MCP_dict = {'RSNR': SNR_mean_list_TF_MCP}
# sensing = 'normal'
# sio.savemat(f'stored_results/TF_MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations', TF_MCP_dict)

# ----------------------------------------------------------------------------------------

# %% analysis RTF MCP

import importlib
importlib.reload(a_algorithms)


device = 'cuda:1'
import time
start = time.time()
x_out, x_list, time_list = a_algorithms.a_RTF_MCP(y, A, D, device, numIter, thr_RTF_MCP, gam_RTF_MCP)
end = time.time()
print(f'time elapsed {round(end - start, 2)} seconds' )

N_test = y.shape[1]; SNR_list = []
for i in range(N_test):
    err = np.linalg.norm(x_out[:, i] - x[:, i])
    RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
    if np.isnan(RSNR):
        break
    SNR_list.append(RSNR)

SNR_list_RTF_MCP = np.array(SNR_list)

# x_out = x_out.cpu().T
err = np.linalg.norm(x_out - x, 'fro')
MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
PES_mean, PES_std = a_algorithms.pes(x, x_out)

print('Testing: my RTF MCP MSE is ', round(MSE, 2))
print(f'Testing: my RTF MCP avg SNR: { np.mean(SNR_list_RTF_MCP):.2f} std: {np.std(SNR_list_RTF_MCP):.2f}')
print(f'Testing: my RTF MCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}')
f.write( '-'* 50 + '\n')
f.write(f'RTF MCP MSE = {round(MSE, 2)} dB \nRTF MCP avg SNR: { np.mean(SNR_list_RTF_MCP):.2f} std: {np.std(SNR_list_RTF_MCP):.2f}\nRTF MCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}\nTime = {round(end - start, 2)} seconds\n')


SNR_mean_list_RTF_MCP = []; RSNR_dict = {}

with torch.no_grad():
    loss= []
    D_t = torch.tensor(A, dtype=(torch.float32))
    for x_b in x_list:
        x_b = x_b.cpu().T
        err = np.linalg.norm(x_b - x, 'fro')
        RSNR = -20*np.log10(np.linalg.norm(x, 'fro')/err)
        SNR_mean_list_RTF_MCP.append(RSNR)

# RTF_MCP_dict = {'RSNR': SNR_mean_list_RTF_MCP}
# sensing = 'normal'
# sio.savemat(f'stored_results/RTF_MCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations', RTF_MCP_dict)

# ----------------------------------------------------------------------------------------

# %% analysis Fast MCP

import importlib
importlib.reload(a_algorithms)


device = 'cuda:1'
import time
start = time.time()
x_out, x_list, time_list = a_algorithms.a_FMCP(y, A, D, device, numIter, thr_FMCP, gam_FMCP)
end = time.time()
print(f'time elapsed {round(end - start, 2)} seconds' )

N_test = y.shape[1]; SNR_list = []
for i in range(N_test):
    err = np.linalg.norm(x_out[:, i] - x[:, i])
    RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
    if np.isnan(RSNR):
        break
    SNR_list.append(RSNR)

SNR_list_FMCP = np.array(SNR_list)

# x_out = x_out.cpu().T
err = np.linalg.norm(x_out - x, 'fro')
MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
PES_mean, PES_std = a_algorithms.pes(x, x_out)

print('Testing: my FMCP MSE is ', round(MSE, 2))
print(f'Testing: my FMCP avg SNR: { np.mean(SNR_list_FMCP):.2f} std: {np.std(SNR_list_FMCP):.2f}')
print(f'Testing: my FMCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}')
f.write( '-'* 50 + '\n')
f.write(f'FMCP MSE = {round(MSE, 2)} dB \nFMCP avg SNR: { np.mean(SNR_list_FMCP):.2f} std: {np.std(SNR_list_FMCP):.2f}\nFMCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}\nTime = {round(end - start, 2)} seconds\n')


SNR_mean_list_FMCP = []; RSNR_dict = {}

with torch.no_grad():
    loss= []
    D_t = torch.tensor(A, dtype=(torch.float32))
    for x_b in x_list:
        x_b = x_b.cpu().T
        err = np.linalg.norm(x_b - x, 'fro')
        RSNR = -20*np.log10(np.linalg.norm(x, 'fro')/err)
        SNR_mean_list_FMCP.append(RSNR)

FMCP_dict = {'RSNR': SNR_mean_list_FMCP}
sensing = 'normal'
sio.savemat(f'stored_results/FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations', FMCP_dict)

# ----------------------------------------------------------------------------------------


# %% analysis Tight Frame Fast MCP

import importlib
importlib.reload(a_algorithms)


device = 'cuda:0'
import time
start = time.time()
x_out, x_list, time_list = a_algorithms.a_TF_FMCP(y, A, D, device, numIter, thr_TF_FMCP, gam_TF_FMCP)
end = time.time()
print(f'time elapsed {round(end - start, 2)} seconds' )

N_test = y.shape[1]; SNR_list = []
for i in range(N_test):
    err = np.linalg.norm(x_out[:, i] - x[:, i])
    RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
    if np.isnan(RSNR):
        break
    SNR_list.append(RSNR)

SNR_list_TF_FMCP = np.array(SNR_list)

# x_out = x_out.cpu().T
err = np.linalg.norm(x_out - x, 'fro')
MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
PES_mean, PES_std = a_algorithms.pes(x, x_out)

print('Testing: my TF FMCP MSE is ', round(MSE, 2))
print(f'Testing: my TF FMCP avg SNR: { np.mean(SNR_list_TF_FMCP):.2f} std: {np.std(SNR_list_TF_FMCP):.2f}')
print(f'Testing: my TF FMCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}')
# f.write( '-'* 50 + '\n')
# f.write(f'TF FMCP MSE = {round(MSE, 2)} dB \nTF FMCP avg SNR: { np.mean(SNR_list_TF_FMCP):.2f} std: {np.std(SNR_list_TF_FMCP):.2f}\nTF FMCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}\nTime = {round(end - start, 2)} seconds\n')


SNR_mean_list_TF_FMCP = []; RSNR_dict = {}

with torch.no_grad():
    loss= []
    D_t = torch.tensor(A, dtype=(torch.float32))
    for x_b in x_list:
        x_b = x_b.cpu().T
        err = np.linalg.norm(x_b - x, 'fro')
        RSNR = -20*np.log10(np.linalg.norm(x, 'fro')/err)
        SNR_mean_list_TF_FMCP.append(RSNR)

TF_FMCP_dict = {'RSNR': SNR_mean_list_TF_FMCP}
sensing = 'normal'
sio.savemat(f'stored_results/TF_FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations', TF_FMCP_dict)

# ----------------------------------------------------------------------------------------


# %% analysis Tight Frame Fast MCP

import importlib
importlib.reload(a_algorithms)


device = 'cuda:0'
import time
start = time.time()
x_out, x_list, time_list = a_algorithms.a_RTF_FMCP(y, A, D, device, numIter, thr_RTF_FMCP, gam_RTF_FMCP)
end = time.time()
print(f'time elapsed {round(end - start, 2)} seconds' )

N_test = y.shape[1]; SNR_list = []
for i in range(N_test):
    err = np.linalg.norm(x_out[:, i] - x[:, i])
    RSNR = 20*np.log10(np.linalg.norm(x[:, i])/err)
    if np.isnan(RSNR):
        break
    SNR_list.append(RSNR)

SNR_list_RTF_FMCP = np.array(SNR_list)

# x_out = x_out.cpu().T
err = np.linalg.norm(x_out - x, 'fro')
MSE = -20*np.log10(np.linalg.norm(x, 'fro')/err)
PES_mean, PES_std = a_algorithms.pes(x, x_out)

print('Testing: my TF FMCP MSE is ', round(MSE, 2))
print(f'Testing: my TF FMCP avg SNR: { np.mean(SNR_list_RTF_FMCP):.2f} std: {np.std(SNR_list_RTF_FMCP):.2f}')
print(f'Testing: my TF FMCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}')
# f.write( '-'* 50 + '\n')
# f.write(f'TF FMCP MSE = {round(MSE, 2)} dB \nTF FMCP avg SNR: { np.mean(SNR_list_TF_FMCP):.2f} std: {np.std(SNR_list_TF_FMCP):.2f}\nTF FMCP avg PES: { PES_mean:.4f} std: {PES_std:.4f}\nTime = {round(end - start, 2)} seconds\n')


SNR_mean_list_RTF_FMCP = []; RSNR_dict = {}

with torch.no_grad():
    loss= []
    D_t = torch.tensor(A, dtype=(torch.float32))
    for x_b in x_list:
        x_b = x_b.cpu().T
        err = np.linalg.norm(x_b - x, 'fro')
        RSNR = -20*np.log10(np.linalg.norm(x, 'fro')/err)
        SNR_mean_list_RTF_FMCP.append(RSNR)

RTF_FMCP_dict = {'RSNR': SNR_mean_list_RTF_FMCP}
sensing = 'normal'
sio.savemat(f'stored_results/RTF_FMCP_iter_{numIter}_noise_{noise_level:.1e}_sparsity_{sparsity}_sensing_{sensing}_vs_iterations', RTF_FMCP_dict)

# ----------------------------------------------------------------------------------------

#%% PLot the results with Plotting tools
from plotting_tools import *
import scienceplots

plt.style.use(['science','ieee'])
plt.rcParams.update({"font.family": "sans","font.serif": ["cm"],"mathtext.fontset": "cm","font.size": 26}) 

ISTA_color = 'green'; TF_color = 'blue'; MCP_color = 'red'; TF_MCP_color = 'm'
width = 2
RTF_MCP_color = 'c';

plt.figure(figsize=(8,6))
ax = plt.gca()

# plot_signal(range(1, 1+len(SNR_mean_list_ISTA)), SNR_mean_list_ISTA, ax=ax,
#     legend_label=r'ISTA',
#     line_width=width, plot_colour=ISTA_color, line_style='-')

plot_signal(range(1, 1+len(SNR_mean_list_MCP)), SNR_mean_list_MCP, ax=ax,
    legend_label=r'MCP',
    line_width=width, plot_colour=MCP_color, line_style='-',
    xlimits = [0, numIter], ylimits = [-20,30])

plot_signal(range(1, 1+len(SNR_mean_list_RTF_MCP)), SNR_mean_list_RTF_MCP, ax=ax,
    legend_label=r'RTF-MCP',
    line_width=width, plot_colour=RTF_MCP_color, line_style='-',
    xlimits = [0, numIter], ylimits = [-20,30])

plot_signal(range(1, 1+len(SNR_mean_list_FMCP)), SNR_mean_list_FMCP, ax=ax,
    legend_label=r'FMCP',
    line_width=width, plot_colour=MCP_color, line_style='--',
    xlimits = [0, numIter], ylimits = [-20,30])

plot_signal(range(1, 1+len(SNR_mean_list_TF_FMCP)), SNR_mean_list_TF_FMCP, ax=ax,
    legend_label=r'TF-FMCP',
    line_width=width, plot_colour=TF_MCP_color, line_style='--',
    xlimits = [0, numIter], ylimits = [-20,30])

plot_signal(range(1, 1+len(SNR_mean_list_RTF_FMCP)), SNR_mean_list_RTF_FMCP, ax=ax,
    legend_label=r'RTF-FMCP',
    line_width=width, plot_colour=RTF_MCP_color, line_style='--',
    xlimits = [0, numIter], ylimits = [-20,30])

# plot_signal(range(1, 1+len(SNR_mean_list_TF_MCP)), SNR_mean_list_TF_MCP, ax=ax,
#     legend_label=r'TF ISTA',
#     line_width=width, plot_colour=TF_color, line_style='-',
#     xlimits = [0, numIter], ylimits = [-20,30])

plot_signal(range(1, 1+len(SNR_mean_list_TF_MCP)), SNR_mean_list_TF_MCP, ax=ax,
    legend_label=r'TF-MCP',
    line_width=width, plot_colour=TF_MCP_color, line_style='-',
    xaxis_label=r'ITERATIONS', yaxis_label=r'MSE [DB]',
    n_col=2, legend_loc='upper right',
    xlimits=[0,numIter], ylimits=[y_lim, 0], 
    save = f'./RTF_FMCP/figures/sparsity_{sparsity}_iter_{numIter}_SNR_{round(SNR,2)}_m_{m}_n_{n}')

f.close()
# plot_signal(range(1, 1+len(SNR_mean_list_RTF_ISTA)), SNR_mean_list_RTF_ISTA, ax=ax,
#     legend_label=r'RTF-ISTA',
#     line_width=width, plot_colour=RTF_color, line_style='-',
#     xaxis_label=r'ITERATIONS', yaxis_label=r'MSE [DB]',
#     n_col=1, legend_loc='upper right',
#     xlimits=[0,numIter], ylimits=[-25, 0], save = f'./figures/sparsity_{sparsity}_noise_{noise_level}_iter_{numIter}_SNR_{SNR}')

# %
# plt.figure(figsize=(8,6))
# ax = plt.gca()

# plot_signal(range(1, 1+len(SNR_mean_list_FISTA)), SNR_mean_list_FISTA, ax=ax,
#     legend_label=r'FISTA',
#     line_width=width, plot_colour=ISTA_color, line_style='--')

# plot_signal(range(1, 1+len(SNR_mean_list_TF_FISTA)), SNR_mean_list_TF_FISTA, ax=ax,
#     legend_label=r'TF-FISTA',
#     line_width=width, plot_colour=TF_color, line_style='--')

# plot_signal(range(1, 1+len(SNR_mean_list_RTF_FISTA)), SNR_mean_list_RTF_FISTA, ax=ax,
#     legend_label=r'RTF-FISTA',
#     line_width=width, plot_colour=RTF_color, line_style='--',
#     xaxis_label=r'ITERATIONS', yaxis_label=r'MSE [DB]',
#     n_col=1, legend_loc='upper right',
#     xlimits=[0,numIter], ylimits=[-25, 0], save = f'./figures/full_sparsity_{sparsity}_noise_{noise_level}_iter_{numIter}_SNR_{SNR}')

# plt.plot( SNR_mean_list_FISTA, label = 'FISTA')
# plt.plot( SNR_mean_list_ISTA, label = 'ISTA')
# plt.plot( SNR_mean_list_TF_ISTA, label = 'TF-ISTA')
# plt.plot( SNR_mean_list_TF_FISTA, label = 'TF-FISTA')
# plt.plot( SNR_mean_list_RTF_ISTA, label = 'RTF-ISTA')
# plt.plot( SNR_mean_list_RTF_FISTA, label = 'RTF-FISTA')
# plt.legend()
# plt.xlabel('iterations')
# plt.ylabel('RSNR')
# # plt.ylim([0, 26])
# plt.savefig('a_loss.pdf')
# %
# plt.figure(figsize=(8,6))
# ax = plt.gca()
# loc = np.argmin(SNR_list_TF_MCP)
# plt.stem(x_b[:, loc], markerfmt='rd')
# plt.stem(x[:, loc], markerfmt='bo')
# plt.show()

# % Testing Soft threshold and Firm threshold proximal operations
# sig = np.arange(-10, 10, 1)

# # plt.style.use(['science','ieee'])
# # plt.rcParams.update({"font.family": "sans","font.serif": ["cm"],"mathtext.fontset": "cm","font.size": 26}) 
# sig_t = torch.tensor(sig)
# soft = soft_thr(sig_t, 2).numpy()
# firm = firm_thr(sig_t, 2, 3).numpy()


# plt.figure(figsize=(8,6))
# ax = plt.gca()
# plt.plot(sig, sig, '-', label='Y=X')
# plt.plot(sig, soft, label='Soft')
# plt.plot(sig, firm, label = 'Firm')
# plt.legend()
# plt.show()

# % Test PES
# import importlib
# importlib.reload(a_algorithms)

# a = np.array([[0,0,0,1,2,3,0,0,0,4], [0,0,0,1,2,3,0,0,0,4]])
# b = np.array([[0,0,0,0,2,3,0,0,0,4], [0,0,0,0,2,3,0,0,0,4]])

# u, v = a_algorithms.pes(a.T, b.T)
# print(u, v)
# %%
