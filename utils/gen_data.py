#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:38:54 2022

@author: alitaghibakhshi
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import os
import os.path
import sys
import torch as T
import copy
import random
from Unstructured import *
import scipy
from grids_gpu import *
import time
from utils_gpu import *
import argparse
from torch_geometric.data import Data, HeteroData
from pyamg.aggregation import lloyd_aggregation

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

data_parser = argparse.ArgumentParser(description='Settings for generating data')

data_parser.add_argument('--directory', type=str, default='../Data/new_data', help='Saving directory')
data_parser.add_argument('--num-data', type=int, default=1, help='Number of generated data')
data_parser.add_argument('--ratio', type=tuple, default=(0.012, 0.03), help='Lower and upper bound for ratio')
data_parser.add_argument('--size-unstructured', type=tuple, default=(0.2, 0.5), help='Lower and upper bound for  unstructured size')
data_parser.add_argument('--hops', type=int, default=1, help='Learnable hops away from boundary')
data_parser.add_argument('--cut', type=int, default=1, help='RAS delta')

data_args = data_parser.parse_args()

def K_means_agg(X, A, ratio):
    
    R, idx = lloyd_aggregation(A, max(ratio, 2/A.shape[0]))
    sum_R = scipy.sparse.diags(1/np.array(R.sum(0))[0])
    new_R = R @ sum_R
    idx = torch.tensor(idx)
    A_coarse = R.transpose() @ A @ new_R
    coarse_index = torch.tensor(np.int64(A_coarse.nonzero()))

    X_coarse = (X.t() @ torch.tensor(new_R.toarray()).float()).t()
    
    fine2coarse = torch.tensor(np.int64(R.nonzero()))
    attr_coarse = torch.tensor(A_coarse.toarray().flatten()[A_coarse.toarray().flatten().nonzero()]).unsqueeze(1).float()
    
    return X_coarse, coarse_index, idx, fine2coarse, attr_coarse, A_coarse, R.transpose()

def K_means_agg_torch(X, A, ratio, grid):
    
    # R, idx = lloyd_aggregation(A, ratio)

    R = grid.R0.transpose()
    
    tR = torch.tensor(R.toarray()).float().to(device)#.float()
    tA = torch.tensor(A.toarray()).float().to(device)#.float()
    idx = grid.aggop[1]
    
    idx = torch.tensor(idx).to(device)
    # A_coarse = R.transpose() @ A @ R
    A_coarse = tR.t() @ tA @ tR
    A_coarse = scipy.sparse.csr_matrix(np.array(A_coarse.cpu()))
    
    coarse_index = torch.tensor(np.int64(A_coarse.nonzero())).to(device)
    # coarse_index = A_coarse.to_dense().nonzero().to(device)
    # coarse_index = coarse_index.reshape(coarse_index.shape[1], coarse_index.shape[0])
    
    X_coarse = (X.t() @ torch.tensor(R.toarray()).float().to(device)).t()
    # X_coarse = (X.t() @ tsR.to_dense()).t()

    
    # fine2coarse = torch.tensor(np.int64(R.nonzero())).to(device)
    
    neigh_R0 = grid.neigh_R0.transpose()
    
    fine2coarse = torch.tensor(np.int64(neigh_R0.nonzero())).to(device)
    
    

    # fine2coarse = tsR.to_dense().nonzero().to(device)
    # fine2coarse = fine2coarse.reshape(fine2coarse.shape[1], fine2coarse.shape[0])
    
    attr_coarse = torch.tensor(A_coarse.toarray().flatten()[A_coarse.toarray().flatten().nonzero()]).unsqueeze(1).float().to(device)
    # attr_coarse = A_coarse.to_dense().flatten()[A_coarse.to_dense().flatten().nonzero()].float().to(device)

    return X_coarse, coarse_index, idx, fine2coarse, attr_coarse, A_coarse, grid.neigh_R0


def make_graph(lvl, grid, ratio):
    
    all_x, all_edge_index, all_edge_attr = grid.gdata.x, grid.gdata.edge_index, grid.gdata.edge_attr.float()
    A = grid.A
    graphs = {}
    
    P = {}
    hetero_edges = {}
    
    x = {}
    edge_index = {}
    edge_attr = {}
    idx = {}
    fine2coarse = {}
    edge_attr = {}
    dict_A = {}
    dict_A[0] = A
    x[0], edge_index[0], edge_attr[0] = all_x, all_edge_index, all_edge_attr
    graphs[0] = Data(x[0], edge_index[0], edge_attr[0])
        
    for i in range(1,lvl):
        
        if i ==1:
            x[i], edge_index[i], idx[i], fine2coarse[i], edge_attr[i], dict_A[i], P[i] = K_means_agg_torch(x[i-1], dict_A[i-1], ratio, grid)
        else:
            x[i], edge_index[i], idx[i], fine2coarse[i], edge_attr[i], dict_A[i], P[i] = K_means_agg(x[i-1], dict_A[i-1], ratio)
            

        graphs[i] = Data(x[i], edge_index[i], edge_attr[i]).to(device)
        
    data = HeteroData()

    
    for i in range(1, lvl):
        for j in range(i):
            
            proj_op = 1
            for k in range(i-j):
            
                proj_op  = proj_op * P[i-k]
                
            proj_op = proj_op.transpose()
            
            hetero_edges = torch.tensor(np.int64(proj_op.nonzero()))
   
   
            data['L'+str(j),'->','L'+str(i)].edge_index = hetero_edges
            data['L'+str(i),'->','L'+str(j)].edge_index = torch.tensor([hetero_edges[1].tolist(), hetero_edges[0].tolist()]).to(device)

    for i in range(lvl):
        
        data['L'+str(i)].x = x[i].to(device)
        data['L'+str(i),'-','L'+str(i)].edge_index = edge_index[i].to(device)
        data['L'+str(i),'-','L'+str(i)].edge_attr = edge_attr[i].to(device)
        
    return data.x_dict, data.edge_index_dict, data.edge_attr_dict
    

def generate_data(data_args, show_fig = False):
    
    path = data_args.directory
    
    if not os.path.exists(path):
        os.makedirs(path)
        

    for i in range(data_args.num_data):
        
        lcmin = 0.1#np.random.uniform(0.105, 0.12)

        lcmax = 0.1#np.random.uniform(0.10, 0.12)
        n = np.random.choice([3,4,5,6,7,8,9,10,20,40])
        randomized = True if np.random.rand() < 0.4 else True
        g = rand_grid_gen1(randomized = randomized, n = n, min_ = 0.03, min_sz = 0.6, 
                      lcmin = lcmin, lcmax = lcmax, distmin = 0.01, distmax = 0.035, PDE = 'Poisson')

        
        num_node = g.num_nodes
        ratio = 1/12 #25*((g.num_nodes/600)**0.5)/g.num_nodes

        grid =  Grid_PWA(g.A, g.mesh, max(2/g.num_nodes, ratio), hops = data_args.hops, 
                          cut=data_args.cut, h = 1, nu = 0, BC = 'Dirichlet') 
        
        lvl = 3
        dict_data = make_graph(lvl, grid, ratio)
        
        num_dom = grid.aggop[0].shape[-1]
            
            
        print("grid number = ", i, ", number of nodes  ", num_node, ", number of domains = ", num_dom)
        
        if show_fig:
            grid.plot_agg(size = 1, labeling = False, w = 0.1,shade = 0.01)
            plt.title (f'Grid nodes = {grid.A.shape[0]}, subdomains = {num_dom}, nodes = {num_node}')
            plt.show()
            
        grid.dict_data = dict_data
        torch.save(grid, path+"/grid"+str(i)+".pth")
    torch.save(data_args, path+"/data_config.pth")
            
generate_data(data_args, show_fig = False)

