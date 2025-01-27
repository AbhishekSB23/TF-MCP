import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import uuid
import torch.nn as nn
import numpy.linalg as LA
import torch.nn.functional as F

def soft_thr(input_, theta_):
    return F.relu(input_-theta_)-F.relu(-input_-theta_)

def firm_thr(input_, theta_, gam_):
    T = theta_*gam_
    return (1/(gam_-1))*F.relu(-input_-T)-(gam_/(gam_-1))*F.relu(-input_-theta_)+(gam_/(gam_-1))*F.relu(input_-theta_)-(1/(gam_-1))*F.relu(input_-T)

def pes(x,x_est):
  d = []
  for i in range(x.shape[1]):
    M = max(np.sum(x[:,i] != 0),np.sum(x_est[:,i] != 0))  
    d.append((M - np.sum((x[:,i]!=0) * (x_est[:,i]!=0)))/M)
  return np.mean(d),np.std(d)

# ----------------------------------------------------------------------------------
# 
#      ````````````````    ISTA Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% FISTA model and algorithm below

class a_ISTA_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, alpha, device):
        super(a_ISTA_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = self.alpha
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(A.T, A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*A.T)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        d_0 = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        alpha_1 = 1
        alpha_0 = 1
        
        for iter in range(self.numIter):
            # d_old = d
            # alpha_0 = alpha_1
            # alpha_1 = 0.5 + np.sqrt(1 + 4 * alpha_0 ** 2)/2
            # z = d + (alpha_0 - 1)/alpha_1 * (d - d_0)
            # d_0 = d
            # d = soft_thr(self._W(y) + self._S(d), self.thr)
            # print(D.shape)
            # print((torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).shape)
            d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-4:
            #     break
        return x, time_list

def a_ISTA(Y, A, D, device, numIter, thr_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_ISTA_model(m, n, A, D, numIter, thr_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list

# ----------------------------------------------------------------------------------
# 
#      ````````````````    FISTA Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% FISTA model and algorithm below

class a_FISTA_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, alpha, device):
        super(a_FISTA_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = self.alpha
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(A.T, A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*A.T)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        d_0 = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        alpha_1 = 1
        alpha_0 = 1
        
        for iter in range(self.numIter):
            d_old = d
            alpha_0 = alpha_1
            alpha_1 = 0.5 + np.sqrt(1 + 4 * alpha_0 ** 2)/2
            z = d + (alpha_0 - 1)/alpha_1 * (d - d_0)
            d_0 = d
            # d = soft_thr(self._W(y) + self._S(d), self.thr)
            # print(D.shape)
            # print((torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).shape)
            d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-4:
            #     break
        return x, time_list

def a_FISTA(Y, A, D, device, numIter, thr_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_FISTA_model(m, n, A, D, numIter, thr_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list

# ----------------------------------------------------------------------------------
# 
#      ````````````````    TF-ISTA Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% FISTA model and algorithm below

class a_TF_ISTA_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, alpha, device):
        super(a_TF_ISTA_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = 1
        Ainv = LA.pinv(A)
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(Ainv, A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*Ainv)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        d_0 = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        alpha_1 = 1
        alpha_0 = 1
        
        for iter in range(self.numIter):
            # d_old = d
            # alpha_0 = alpha_1
            # alpha_1 = 0.5 + np.sqrt(1 + 4 * alpha_0 ** 2)/2
            # z = d + (alpha_0 - 1)/alpha_1 * (d - d_0)
            # d_0 = d
            # d = soft_thr(self._W(y) + self._S(d), self.thr)
            # print(D.shape)
            # print((torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).shape)
            d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-4:
            #     break
        return x, time_list

def a_TF_ISTA(Y, A, D, device, numIter, thr_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_TF_ISTA_model(m, n, A, D, numIter, thr_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list

# ----------------------------------------------------------------------------------
# 
#      ````````````````    TF-FISTA Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% FISTA model and algorithm below

class a_TF_FISTA_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, alpha, device):
        super(a_TF_FISTA_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = 1
        Ainv = LA.pinv(A)
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(Ainv, A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*Ainv)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        d_0 = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        alpha_1 = 1
        alpha_0 = 1
        
        for iter in range(self.numIter):
            d_old = d
            alpha_0 = alpha_1
            alpha_1 = 0.5 + np.sqrt(1 + 4 * alpha_0 ** 2)/2
            z = d + (alpha_0 - 1)/alpha_1 * (d - d_0)
            d_0 = d
            # d = soft_thr(self._W(y) + self._S(d), self.thr)
            # print(D.shape)
            # print((torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).shape)
            d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-4:
            #     break
        return x, time_list

def a_TF_FISTA(Y, A, D, device, numIter, thr_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_TF_FISTA_model(m, n, A, D, numIter, thr_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list

# ----------------------------------------------------------------------------------
# 
#      ````````````````    RTF-ISTA Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% FISTA model and algorithm below

class a_RTF_ISTA_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, alpha, device):
        super(a_RTF_ISTA_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        Ainv = LA.pinv(A)
        AinvA = Ainv @ A
        d_AinvA = LA.inv(np.diag(np.diag(AinvA)))
        Ainv = d_AinvA @ Ainv
        alpha = LA.norm(Ainv @ A, 2) * 1.001

        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(Ainv, A))
        # print(np.diag(Ainv @ A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*Ainv)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        d_0 = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        alpha_1 = 1
        alpha_0 = 1
        
        for iter in range(self.numIter):
            # d_old = d
            # alpha_0 = alpha_1
            # alpha_1 = 0.5 + np.sqrt(1 + 4 * alpha_0 ** 2)/2
            # z = d + (alpha_0 - 1)/alpha_1 * (d - d_0)
            # d_0 = d
            # d = soft_thr(self._W(y) + self._S(d), self.thr)
            # print(D.shape)
            # print((torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).shape)
            d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-4:
            #     break
        return x, time_list

def a_RTF_ISTA(Y, A, D, device, numIter, thr_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_RTF_ISTA_model(m, n, A, D, numIter, thr_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list

# ----------------------------------------------------------------------------------
# 
#      ````````````````    RTF-FISTA Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% FISTA model and algorithm below

class a_RTF_FISTA_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, alpha, device):
        super(a_RTF_FISTA_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        Ainv = LA.pinv(A)
        AinvA = Ainv @ A
        d_AinvA = LA.inv(np.diag(np.diag(AinvA)))
        Ainv = d_AinvA @ Ainv
        alpha = LA.norm(Ainv @ A, 2) * 1.001

        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(Ainv, A))
        # print(np.diag(Ainv @ A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*Ainv)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))


    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        d_0 = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        alpha_1 = 1
        alpha_0 = 1
        
        for iter in range(self.numIter):
            d_old = d
            alpha_0 = alpha_1
            alpha_1 = 0.5 + np.sqrt(1 + 4 * alpha_0 ** 2)/2
            z = d + (alpha_0 - 1)/alpha_1 * (d - d_0)
            d_0 = d
            # d = soft_thr(self._W(y) + self._S(d), self.thr)
            # print(D.shape)
            # print((torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).shape)
            d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-4:
            #     break
        return x, time_list

def a_RTF_FISTA(Y, A, D, device, numIter, thr_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_RTF_FISTA_model(m, n, A, D, numIter, thr_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list


# ----------------------------------------------------------------------------------
# 
#      ````````````````    MCP Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% MCP model and algorithm below

class a_MCP_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, gam_, alpha, device):
        super(a_MCP_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.gam = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.g = gam_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = self.alpha
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(A.T, A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*A.T)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        gam = torch.ones(1, 1) * self.g
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))
        self.gam.data = nn.Parameter(gam.to(self.device))



    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        for iter in range(self.numIter):
            d_old = d
            d = torch.mm(D.T, firm_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr, self.gam)).T
            # d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-5:
            #     break
        return x, time_list

def a_MCP(Y, A, D, device, numIter, thr_, gam_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_MCP_model(m, n, A, D, numIter, thr_, gam_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list


# ----------------------------------------------------------------------------------
# 
#      ````````````````   TF MCP Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% MCP model and algorithm below

class a_TF_MCP_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, gam_, alpha, device):
        super(a_TF_MCP_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.gam = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.g = gam_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = 1
        Ainv = LA.pinv(A)
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(Ainv, A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*Ainv)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        gam = torch.ones(1, 1) * self.g
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))
        self.gam.data = nn.Parameter(gam.to(self.device))



    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        for iter in range(self.numIter):
            d_old = d
            d = torch.mm(D.T, firm_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr, self.gam)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-5:
            #     break
        return x, time_list

def a_TF_MCP(Y, A, D, device, numIter, thr_, gam_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_TF_MCP_model(m, n, A, D, numIter, thr_, gam_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list

# ----------------------------------------------------------------------------------
# 
#      ````````````````   RTF MCP Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% RTF MCP model and algorithm below

class a_RTF_MCP_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, gam_, alpha, device):
        super(a_RTF_MCP_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.gam = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.g = gam_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        Ainv = LA.pinv(A)
        AinvA = Ainv @ A
        d_AinvA = LA.inv(np.diag(np.diag(AinvA)))
        Ainv = d_AinvA @ Ainv
        alpha = LA.norm(Ainv @ A, 2) * 1.001

        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(Ainv, A))
        # print(np.diag(Ainv @ A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*Ainv)
        B = B.float().to(self.device)

        thr = torch.ones(1, 1) * self.t / alpha
        gam = torch.ones(1, 1) * self.g
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))
        self.gam.data = nn.Parameter(gam.to(self.device))



    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)
        
        for iter in range(self.numIter):
            d_old = d
            d = torch.mm(D.T, firm_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr, self.gam)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-5:
            #     break
        return x, time_list

def a_RTF_MCP(Y, A, D, device, numIter, thr_, gam_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_RTF_MCP_model(m, n, A, D, numIter, thr_, gam_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list



# ----------------------------------------------------------------------------------
# 
#      ````````````````    Fast MCP Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% FMCP model and algorithm below

class a_FMCP_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, gam_, alpha, device):
        super(a_FMCP_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.gam = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.g = gam_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = self.alpha
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(A.T, A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*A.T)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        gam = torch.ones(1, 1) * self.g
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))
        self.gam.data = nn.Parameter(gam.to(self.device))



    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)

        alpha_1 = 1
        alpha_0 = 1
        d_old = d
        
        for iter in range(self.numIter):
            
            alpha_0 = alpha_1
            alpha_1 = 0.5 + np.sqrt(1 + 4 * alpha_0 ** 2)/2
            z = d + (alpha_0 - 1)/alpha_1 * (d - d_old)
            d_old = d
            d = torch.mm(D.T, firm_thr(torch.mm(D, (torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).T), self.thr, self.gam)).T
            # d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-5:
            #     break
        return x, time_list

def a_FMCP(Y, A, D, device, numIter, thr_, gam_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_FMCP_model(m, n, A, D, numIter, thr_, gam_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list


# ----------------------------------------------------------------------------------
# 
#      ````````````````    TF Fast MCP Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% TF_FMCP model and algorithm below

class a_TF_FMCP_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, gam_, alpha, device):
        super(a_TF_FMCP_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.gam = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.g = gam_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        alpha = 1
        Ainv = LA.pinv(A)
        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(Ainv, A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*Ainv)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        gam = torch.ones(1, 1) * self.g
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))
        self.gam.data = nn.Parameter(gam.to(self.device))



    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)

        alpha_1 = 1
        alpha_0 = 1
        d_old = d
        
        for iter in range(self.numIter):
            
            alpha_0 = alpha_1
            alpha_1 = 0.5 + np.sqrt(1 + 4 * alpha_0 ** 2)/2
            z = d + (alpha_0 - 1)/alpha_1 * (d - d_old)
            d_old = d
            d = torch.mm(D.T, firm_thr(torch.mm(D, (torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).T), self.thr, self.gam)).T
            # d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-5:
            #     break
        return x, time_list

def a_TF_FMCP(Y, A, D, device, numIter, thr_, gam_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_TF_FMCP_model(m, n, A, D, numIter, thr_, gam_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list



# ----------------------------------------------------------------------------------
# 
#      ````````````````    RTF Fast MCP Module     !!!!!!!!!!!!!!!
# 
# ----------------------------------------------------------------------------------
#%% RTF_FMCP model and algorithm below

class a_RTF_FMCP_model(nn.Module):
    def __init__(self, m, n, A, D, numIter, thr_, gam_, alpha, device):
        super(a_RTF_FMCP_model, self).__init__()
        self._W = nn.Linear(in_features = m, out_features = n, bias=False)
        self._S = nn.Linear(in_features = n, out_features = n,
                            bias=False)

        self.thr = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.gam = nn.Parameter(torch.rand(1,1), requires_grad=True)
        self.numIter = numIter
        self.A = A
        self.alpha = alpha
        self.device = device
        self.t = thr_
        self.g = gam_
        self.D2 = D
        
        
    # custom weights initialization called on network
    def weights_init(self):
        A = self.A
        Ainv = LA.pinv(A)
        AinvA = Ainv @ A
        d_AinvA = LA.inv(np.diag(np.diag(AinvA)))
        Ainv = d_AinvA @ Ainv
        alpha = LA.norm(Ainv @ A, 2) * 1.001

        S = torch.from_numpy(np.eye(A.shape[1]) - (1/alpha)*np.matmul(Ainv, A))
        # print(np.diag(Ainv @ A))
        
        D = torch.from_numpy(self.D2)
        self.D = D.float().to(self.device)
        S = S.float().to(self.device)
        B = torch.from_numpy((1/alpha)*Ainv)
        B = B.float().to(self.device)
        
        thr = torch.ones(1, 1) * self.t / alpha
        gam = torch.ones(1, 1) * self.g
        
        self._S.weight = nn.Parameter(S)
        self._W.weight = nn.Parameter(B)
        self.S = S
        self.W = B
        self.thr.data = nn.Parameter(thr.to(self.device))
        self.gam.data = nn.Parameter(gam.to(self.device))



    def forward(self, y):
        D = self.D
        x = []
        d = torch.zeros(y.shape[0], self.A.shape[1], device = self.device)
        
        import time
        start = time.time()
        time_list = []
        # time_list.append(time.time() - start)
        x.append(d)

        alpha_1 = 1
        alpha_0 = 1
        d_old = d
        
        for iter in range(self.numIter):
            
            alpha_0 = alpha_1
            alpha_1 = 0.5 + np.sqrt(1 + 4 * alpha_0 ** 2)/2
            z = d + (alpha_0 - 1)/alpha_1 * (d - d_old)
            d_old = d
            d = torch.mm(D.T, firm_thr(torch.mm(D, (torch.mm(self.S, z.T).T + torch.mm(self.W, y.T).T).T), self.thr, self.gam)).T
            # d = torch.mm(D.T, soft_thr(torch.mm(D, (torch.mm(self.S, d.T).T + torch.mm(self.W, y.T).T).T), self.thr)).T
            x.append(d)
            
            time_list.append(time.time() - start)
            # if torch.norm(d - d_old) < 1e-5:
            #     break
        return x, time_list

def a_RTF_FMCP(Y, A, D, device, numIter, thr_, gam_):

    alpha = (LA.norm(A, 2) ** 2) * 1.001
    m, n = A.shape
    net = a_RTF_FMCP_model(m, n, A, D, numIter, thr_, gam_, alpha, device)
    net.weights_init()
    # convert the data into tensors
    Y_t = torch.from_numpy(Y.T)
    if len(Y.shape) <= 1:
        Y_t = Y_t.view(1, -1)
    Y_t = Y_t.float().to(device)
    D_t = torch.from_numpy(A.T)
    D_t = D_t.float().to(device)

    ratio = 1
    with torch.no_grad():
        # Compute the output
        net.eval()
        X_list, time_list = net(Y_t.float())
        if len(Y.shape) <= 1:
            X_list = X_list.view(-1)
        X_final = X_list[-1].cpu().numpy()
        X_final = X_final.T

    return X_final, X_list, time_list