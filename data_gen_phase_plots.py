import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from a_algorithms import pes
from scipy.fft import dct
import scipy
import scienceplots
import scipy.io as sio


# --------------- Training Data storage -----------------
m = 50; n = 100; Ex = 100 ; D = np.eye(n)
seed = 55; rng = np.random.RandomState(seed)
A = rng.normal(0, 1/np.sqrt(m), [m, n])
A /= LA.norm(A, 2, axis = 0)
# sio.savemat(f'phase_plot_sim_data_train/A', {'A': A})
sio.savemat(f'phase_plot_sim_data_test/A', {'A': A})

# for sparsity in range(10,31):
#     for SNR in range(10,51):

#         al = rng.normal(0, 1, (n, Ex)) * rng.binomial(1, sparsity/100, (n, Ex))

#         for i in range(Ex):
#             if LA.norm(al[:, i]) < 1e-2 :
#                 print('0 norm sparse code')

#         x = D.T @ al
#         y1 = A @ x 
#         noise = rng.normal(0, 1, y1.shape)

#         noise_level = LA.norm(y1, 2) / ((10 ** (SNR / 20)) * LA.norm(noise, 2))

#         y = y1 + noise_level * noise

#         sio.savemat(f'phase_plot_sim_data_train/sparsity_{sparsity}_noise_{SNR}', {'y': y, 'x': x})

# print('train_data_saved')
# --------------- Test Data storage -----------------

m = 50; n = 100; Ex = 1000 ; D = np.eye(n)
seed = 91; rng = np.random.RandomState(seed)

for sparsity in range(10,31):
    print(f'test_data_saved {sparsity}')
    for SNR in range(10,51):

        al = rng.normal(0, 1, (n, Ex)) * rng.binomial(1, sparsity/100, (n, Ex))

        for i in range(Ex):
            if LA.norm(al[:, i]) < 1e-2 :
                print('0 norm sparse code')

        x = D.T @ al
        y1 = A @ x 
        noise = rng.normal(0, 1, y1.shape)

        noise_level = LA.norm(y1, 2) / ((10 ** (SNR / 20)) * LA.norm(noise, 2))

        y = y1 + noise_level * noise

        sio.savemat(f'phase_plot_sim_data_test/sparsity_{sparsity}_noise_{SNR}', {'y': y, 'x': x})


