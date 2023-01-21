# import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
import os
# import fem
import sys
import torch as T
import copy
import random
from torch.utils.tensorboard import SummaryWriter
import scipy
from grids_gpu import *
import time
# mpl.rcParams['figure.dpi'] = 300
import numpy as np
import scipy as sp
from pyamg import amg_core
import numml.sparse as spml

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    

def get_Li (L, grid):
    
    L_i = {}


    for i in range(grid.aggop[0].shape[-1]):

        nz = grid.list_cut_aggs[i]

        learnables = grid.learn_nodes[i]

        
        L_i[i] = torch.zeros(len(nz),len(nz)).double().to(device)

        list_idx = []

        for l in learnables:
            list_idx.append(nz.index(l))
        
        L_i[i][np.ix_(list_idx, list_idx)] = L[np.ix_(learnables, learnables)]
        

    return L_i

softmax = torch.nn.Softmax(dim=0)

def make_sparse_torch(A, sparse = True):
    if sparse:
        idxs = torch.tensor(np.array(A.nonzero()))
        dat = torch.tensor(A.data)
    else:
        idxs = torch.tensor([[i//A.shape[1] for i in range(A.shape[0]*A.shape[1])], 
                             [i% A.shape[1] for i in range(A.shape[0]*A.shape[1])]])
        dat = A.flatten()
    s = torch.sparse_coo_tensor(idxs, dat, (A.shape[0], A.shape[1]))
    return s#.to_sparse_csr()
    
def preconditioner(grid, output, precond_type = False, u = None):
        
    M = 0
    # tsA = torch.tensor(grid.A.toarray()).to(device)
    tsA = spml.SparseCSRTensor(make_sparse_torch(grid.A)).to(device)
    # print (tq5-tq4)
    if precond_type == 'AS':  
        for i in range(grid.aggop[0].shape[-1]):
            
            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()))
            A_inv = torch.linalg.pinv(torch.tensor(grid.R[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R[i].transpose().toarray()))
            M += torch.tensor(grid.R[i].transpose().toarray()) @ A_inv @ torch.tensor(grid.R[i].toarray())
        M = torch.tensor(M)
                
    elif precond_type == 'RAS':

        for i in range(grid.aggop[0].shape[-1]):

            # A_inv = torch.linalg.pinv(torch.tensor(grid.A_i[i].toarray()) + grid.h*L[i])
            A_inv = torch.linalg.pinv((make_sparse_torch(grid.R_hop[i]).to_sparse_csr() @ make_sparse_torch(grid.A).to_sparse_csr() @ make_sparse_torch(grid.R_hop[i]).to_sparse_csr().t()).to_dense())

            # A_inv = torch.linalg.pinv(torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray()))
            
            r0 = grid.R[i].nonzero()[-1].tolist()

            rdelta = grid.R_hop[i].nonzero()[-1].tolist()
            list_ixs = []

            for e in r0:
                list_ixs.append(rdelta.index(e))

            modified_R_i = grid.modified_R[i]
            ####
            
            a1 = scipy.sparse.coo_matrix(modified_R_i).transpose()
            a2 = scipy.sparse.coo_matrix(grid.R_hop[i])
            
            rows = a1.row.tolist()
            cols = a2.col.tolist()
            row_idx = []
            col_idx = []
            for r in rows:
                for _ in range(len(cols)):
                    row_idx.append(r)
                    
            for _ in range(len(rows)):       
                for c in cols:
                    col_idx.append(c)
                    

            list_ixs.sort()
            
            add_M = torch.sparse_coo_tensor(torch.tensor([row_idx, col_idx]), A_inv[list_ixs, :].flatten(), (grid.A.shape[0], grid.A.shape[1]))
            # add_M = torch.zeros(grid.A.shape).double()
            # add_M[np.ix_(rows, cols)] = A_tilde_inv[list_ixs, :]
            #torch.tensor(modified_R_i.transpose()) @ A_tilde_inv @ torch.tensor(grid.R_hop[i].toarray())
            if i == 0:
                M = add_M
            else:
                M += add_M
                
    
    elif precond_type == 'ML_ORAS':
        

        L = get_Li (output, grid)

        for i in range(grid.aggop[0].shape[-1]):
            
            r0 = grid.R[i].nonzero()[-1].tolist()
            rdelta = grid.R_hop[i].nonzero()[-1].tolist()
            list_ixs = []
            for e in r0:
                list_ixs.append(rdelta.index(e))
                
            modified_R_i = grid.modified_R[i]
            

            modified_L = L[i].to(device)
            grid_Rhop_i = spml.SparseCSRTensor(make_sparse_torch(grid.R_hop[i])).to(device)
            # grid_Rhop_i = make_sparse_torch(grid.R_hop[i]).to_dense().to(device)

            AA =  grid_Rhop_i @ tsA @ grid_Rhop_i.T  ####SPSPMM

            # AA = torch.tensor(grid.R_hop[i].toarray()) @ torch.tensor(grid.A.toarray()) @ torch.tensor(grid.R_hop[i].transpose().toarray())
            A_tilde_inv = torch.linalg.pinv(AA.to_dense() + (1/(grid.h**2))*modified_L)
            # add_M = make_sparse_torch(scipy.sparse.csr_matrix(modified_R_i)).t() @ make_sparse_torch(A_tilde_inv, False) @ make_sparse_torch(grid.R_hop[i])
            # M += add_M.to_dense()
            
            a1 = scipy.sparse.coo_matrix(modified_R_i).transpose()
            a2 = scipy.sparse.coo_matrix(grid.R_hop[i])
            
            rows = a1.row.tolist()
            cols = a2.col.tolist()
            row_idx = []
            col_idx = []
            for r in rows:
                for _ in range(len(cols)):
                    row_idx.append(r)
                    
            for _ in range(len(rows)):       
                for c in cols:
                    col_idx.append(c)
                    

            list_ixs.sort()
            
            add_M = torch.sparse_coo_tensor(torch.tensor([row_idx, col_idx]).to(device), A_tilde_inv[list_ixs, :].flatten(), (grid.A.shape[0], grid.A.shape[1])).to(device)

            if i == 0:
                M = add_M
            else:
                M += add_M
            
        
    else:
        raise RuntimeError('Wrong type for preconditioner: '+str(precond_type))
    
    return M
        
        

    
def R0_PoU(grid):
    
    num_nodes  = grid.aggop[0].shape[0]
    num_aggs = grid.aggop[0].shape[-1]
    R0 = np.zeros((num_aggs, num_nodes))
    for i in range(grid.aggop[0].shape[-1]):
        nzs = grid.R_hop[i].nonzero()[-1].tolist()
        R0[i][nzs] = 1
        
    return R0/R0.sum(0)
        
        
def stationary(grid, out, u = None, K = None, precond_type = 'ML_ORAS'):

    M, _ = preconditioner(grid, out, train = True, precond_type = precond_type, u = u)
    
    eprop = M
    
    list_l2 = []
    out_lmax = copy.deepcopy(u)
    for k in range(K):
        out_lmax = eprop @ out_lmax
        l2 = torch.norm(out_lmax, p='fro', dim = 0)
        list_l2.append(l2)
    
    conv_fact = list_l2[-1]#(list_l2[-1]/list_l2[-3])**0.5
    L_max = torch.dot(softmax(conv_fact), conv_fact)

    return L_max
        
def stationary_cycle(A, M, R0, err):

    
    if type(A) == torch.Tensor:
        R0_transpose = R0.t().to(device)
        A0 = R0 @ A @ R0_transpose
        if train:
            A0_inv = torch.linalg.pinv(A0).to(device)
        else:
            A0_inv = torch.linalg.pinv(A0.to_dense()).to(device)

    if type(A) == scipy.sparse.csr.csr_matrix:
        
        R0_transpose = R0.transpose()#.to(device)
        A0 = R0 @ A @ R0_transpose
        A0_inv = np.linalg.pinv(A0.toarray())#.to(device)
    
    if type(A) == numml.sparse.SparseCSRTensor:
        
        A0 = R0 @ A @ R0.T
        A0_inv = torch.linalg.pinv(A0.to_dense()).to(device)
        A0_inv = spml.SparseCSRTensor(A0_inv).to(device)
        
    e = err
    e = A @ e
    e = M @ e
    e = err - e
    err_1 = e
    e = A @ e
    e = R0 @ e
    e = A0_inv @ e
    e = R0.T @ e
    e = err_1 - e
    
    return e



def stationary_max(grid, out, u = None, K = None, precond_type = 'ML_ORAS'):

    M = preconditioner(grid, out, precond_type = precond_type, u = u).to_dense()

    list_l2 = []
    
    out_lmax = spml.SparseCSRTensor(copy.deepcopy(u)).to(device)
    list_max = torch.zeros(K).to(device)
    tsA = spml.SparseCSRTensor(make_sparse_torch(grid.A)).to(device)
    R0 = spml.SparseCSRTensor(out[1])

    for k in range(K):

        out_lmax = stationary_cycle(tsA, M, R0, out_lmax) #+ out_lmax*1e-2
        
        l2 = torch.norm(out_lmax, p='fro', dim = 0)

        list_max[k] = max(l2) ** (1/(k+1))
        list_l2.append(l2)

    L_max = (torch.softmax(list_max, dim = 0) * list_max).sum()#max(list_max)

    mloras_Pcol_norm = 0
    ras_Pcol_norm = 0
    
    r0 = torch.tensor(grid.R0.toarray()).to(device)

    for i in range(grid.R0.shape[0]):
        
        mloras_Pcol_norm += R0[i] @ tsA @ R0.T[:,i]
        ras_Pcol_norm += r0[i] @ tsA @ r0.t()[:,i]
    
    pcol_loss = mloras_Pcol_norm/ras_Pcol_norm
    
    return L_max + pcol_loss*5.0





def torch_2_scipy_sparse(A):
    
    data = A.coalesce().values()
    row = A.coalesce().indices()[0]
    col = A.coalesce().indices()[1]
    out = scipy.sparse.csr_matrix((data, (row, col)), shape=(A.shape[0], A.shape[1]))
    
    return out
    
      
# def test_stationary(grid, output, precond_type, u, K, M=None):

#     if M is None:
#         M = preconditioner(grid, output, train = False, precond_type = precond_type, u = u)
      
#     # eprop_a = M
    
#     out = copy.deepcopy(u.numpy())
#     l2_list = []
#     l2 = np.linalg.norm(out, axis = 0)
#     l2_list.append(max(l2))
#     tsA = grid.A
#     M = torch_2_scipy_sparse(M)
    
#     if precond_type == 'ML_ORAS':

#         R0 = torch_2_scipy_sparse(output[1])
#     else:
#         R0 = grid.R0#grid.aggop[0].transpose()#
#         # R0 = grid.neigh_R0
#     for k in range(K):
#         # out = eprop_a @ out
#         out = stationary_cycle(tsA, M, R0, out)

#         l2 = np.linalg.norm(out, axis = 0)
#         l2_list.append(max(l2))

#     return l2_list

def test_stationary(grid, output, precond_type, u, K, M):

    
    out = copy.deepcopy(u).to(device)
    l2_list = []
    vec_list = []
    l2 = torch.norm(out, p='fro', dim = 0) #np.linalg.norm(out, axis = 0)
    l2_list.append(max(l2))
    vec_list.append(out[:,np.argmax(l2)])
    tsA = make_sparse_torch(grid.A).to_sparse_csr().to(device)
    M = M.to_sparse_csr().to(device) #torch_2_scipy_sparse(M)
    
    if precond_type == 'ML_ORAS':

        R0 = output[1].to_sparse_csr().to(device)

    else:
        R0 = make_sparse_torch(grid.R0).to_sparse_csr().to(device)#grid.aggop[0].transpose()#
        # R0 = make_sparse_torch(grid.neigh_R0).to_sparse_csr().to(device) 

    for k in range(K):
        # out = eprop_a @ out
        out = stationary_cycle(tsA, M, R0, out,train = False)

        l2 = torch.norm(out, p='fro', dim = 0)
        l2_list.append(max(l2))
        vec_list.append(out[:,np.argmax(l2)])


    return l2_list, vec_list

def struct_agg_PWA(n_row, n_col, agg_row, agg_col):


    arg0 = 0
    arg2 = []
    d = int(n_col/agg_col)
    
    for i in range(n_row * n_col):
        
        j = i%n_col
        k = i//n_col
        arg2.append(int(j//agg_col) + (k//agg_row)*d)
        
        
    arg0 = scipy.sparse.csr_matrix((np.ones(n_row * n_col), ([i for i in range(n_row * n_col)], arg2)), 
                                    shape=(n_row * n_col, max(arg2)+1))
            
    arg1 = np.zeros(max(arg2)+1)
    
    return (arg0, arg1, np.array(arg2))



