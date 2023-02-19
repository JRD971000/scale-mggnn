#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 17:07:22 2022

@author: alitaghibakhshi
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch.nn import Linear, ReLU
import os
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.pool import SAGPooling, TopKPooling, ASAPooling, MemPooling, PANPooling
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import subgraph
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear,\
                                 GraphUNet, TAGConv, MessagePassing, NNConv
import sys
import scipy
import torch_geometric
from torch_geometric.data import HeteroData
import torch.optim as optim
from pyamg.aggregation import lloyd_aggregation
from torch.nn.functional import relu, sigmoid
from torch_geometric.nn.norm import PairNorm
from NNs import EdgeModel
import copy
import numml.sparse as spml
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
    
def make_sparse_torch(A, sparse = True):
    if sparse:
        A = scipy.sparse.coo_matrix(A)
        idxs = torch.tensor(np.array(A.nonzero()))
        dat = torch.tensor(A.data)
    else:
        idxs = torch.tensor([[i//A.shape[1] for i in range(A.shape[0]*A.shape[1])], 
                             [i% A.shape[1] for i in range(A.shape[0]*A.shape[1])]])
        dat = A.flatten()
    s = torch.sparse_coo_tensor(idxs, dat, (A.shape[0], A.shape[1]))
    return s#.to_sparse_csr()


class PlainMP(MessagePassing):
    def __init__(self, dim_embed, aggr = 'add'):
        super().__init__(aggr=aggr) #  "Max" aggregation.
        self.net = torch.nn.Sequential(Linear(2*dim_embed, dim_embed), 
                                # torch.nn.ReLU(), Linear(dim_embed, dim_embed),
                                torch.nn.ReLU(), Linear(dim_embed, dim_embed))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        
        x = torch.cat((x_i, x_j), dim = 1)
        x = self.net(x)
        return x#x_i



class MGGNN(torch.nn.Module):
    def __init__(self, lvl, dim_embed, num_layers, K, ratio, lr, PN=False):
        super().__init__()
        self.num_layers = num_layers
        self.PN = PN
        self.lvl = lvl

        # self.droupout = droupout
        self.ratio = ratio
        
        self.pre_edge_main = torch.nn.Sequential(Linear(2, dim_embed), ReLU(), 
                                    Linear(dim_embed, dim_embed), ReLU(),
                                    Linear(dim_embed, 1))
        
        self.pre_edge = torch.nn.Sequential(Linear(1, dim_embed), ReLU(), 
                                    Linear(dim_embed, dim_embed), ReLU(),
                                    Linear(dim_embed, 1))
        
        self.pre_node = torch.nn.Sequential(Linear(1, dim_embed), ReLU(), 
                                    Linear(dim_embed, dim_embed), ReLU(),
                                    Linear(dim_embed, dim_embed))
        
        self.pn = PairNorm()
        self.convs = torch.nn.ModuleList()

        self.convs_fc = torch.nn.ModuleDict()
        for i in range(lvl):
            self.convs_fc[str(i)] = torch.nn.ModuleList()
        self.name  = 'mggnn_lvl'+str(lvl)+'_numlayer'+str(num_layers)
        normalizations = []

        
        for _ in range(num_layers):
            
            dict_ff = {}

            for i in range(lvl):
                dict_fc = {}
                for j in range(lvl):
                    if i==j:
                        dict_ff[('L'+str(i), '-', 'L'+str(i))] = TAGConv(self.lvl*dim_embed, dim_embed, K = K, normalize = False)#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)
                        dict_fc[('L'+str(i), '-', 'L'+str(i))] = TAGConv(dim_embed, dim_embed, K = K, normalize = False)#GATConv(dim_embed, dim_embed, edge_dim = dim_embed)

                    else:
                        dict_fc[('L'+str(i),'->','L'+str(j))] = PlainMP(dim_embed)
                              
                self.convs_fc[str(i)].append(HeteroConv(dict_fc, aggr='add'))            
            
            self.convs.append(HeteroConv(dict_ff, aggr='add'))

            
            normalizations.append(torch_geometric.nn.norm.InstanceNorm(dim_embed))
            
        self.linear_out  = Linear(dim_embed, dim_embed)
        self.linear_out_coarse  = Linear(dim_embed, dim_embed)
        
        self.normaliz = normalizations
        
        self.edge_model  = EdgeModel(dim_embed*2, [dim_embed, int(dim_embed/2), int(dim_embed/4)], 1)
        self.edge_model_R  = EdgeModel(dim_embed*2, [dim_embed, int(dim_embed/2), int(dim_embed/4)], 1)
        
        # self.edge_model_R_dense = torch.nn.Sequential(Linear(dim_embed, dim_embed), 
        #                         torch.nn.ReLU(), Linear(dim_embed, dim_embed),
        #                         torch.nn.ReLU(), Linear(dim_embed, 29))

        # self.network = torch_geometric.nn.Sequential(self.convs)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-5)
        # self.optimizer = optim.RMSprop(self.parameters(), lr = lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.device = torch.device('cpu')
        self.to(self.device)
        
        
    def forward(self, grid, train):
        
        x_dict, edge_index_dict,  edge_attr_dict = copy.deepcopy(grid.dict_data)
        
        for key in x_dict.keys():

            x_dict[key] = x_dict[key].to(device)
            
        for key in edge_index_dict.keys():

            edge_index_dict[key] = edge_index_dict[key].to(device)
            
        for key in edge_attr_dict.keys():

            edge_attr_dict[key] = edge_attr_dict[key].to(device)
        

        edge_attr_dict[('L0', '-', 'L0')] = self.pre_edge_main(edge_attr_dict[('L0', '-', 'L0')])
        
        for key in x_dict.keys():

            x_dict[key] = self.pre_node(x_dict[key])

        for key in edge_attr_dict.keys():
            edge_attr_dict[key] = self.pre_edge(edge_attr_dict[key])

        x_ff = {key: x for key, x in x_dict.items()}

        for i in range(self.num_layers):
            x_out_dict = {}
            
            for j in range(self.lvl) :
                x_out_dict[j] = self.convs_fc[str(j)][i](x_ff, edge_index_dict)#, edge_attr_dict)

                for key in x_out_dict[j].keys():

                    x_out_dict[j][key] = x_out_dict[j][key].relu()
                    #############PiarNorm***************
                    if self.PN:
                        x_out_dict[j][key] = self.pn(x_out_dict[j][key])
                    ###################################
                    
            x_ff_new = {}

            for j in range(self.lvl):

                x_ff_new['L'+str(j)] = torch.cat(tuple([x_out_dict[k]['L'+str(j)] for k in range(self.lvl)]), dim = 1)


            x_ff = x_ff_new

            x_ff = self.convs[i](x_ff, edge_index_dict, edge_attr_dict)

            for key in x_ff.keys():

                x_ff[key] = self.normaliz[i](x_ff[key].relu())
                #############PiarNorm***************
                if self.PN:
                    x_ff[key] = self.pn(x_ff[key])
                ###################################
            
            
        x_coarse = self.linear_out_coarse(x_ff['L1'])
        x = self.linear_out(x_ff['L0'])
        
        row_coarse = edge_index_dict[('L1', '->', 'L0')][0].tolist()
        col_fine = edge_index_dict[('L1', '->', 'L0')][1].tolist()
        
        
        
        
        edge_attr_R = self.edge_model_R(x_coarse[row_coarse], x[col_fine])
        
        out_R = torch.sparse_coo_tensor([row_coarse, col_fine], edge_attr_R.flatten(),
                                                      (grid.R0.shape[0], grid.R0.shape[1])).double()
        
        
        
       
        
        # if train:
        
        sum_row_mat = 1/torch.sparse.sum(out_R, dim = 0).coalesce().values()
        sum_row_mat = spml.SparseCSRTensor(torch.sparse_coo_tensor([np.arange(sum_row_mat.shape[0]).tolist(),
                                                  np.arange(sum_row_mat.shape[0]).tolist()], sum_row_mat, 
                                                  (sum_row_mat.shape[0], sum_row_mat.shape[0])).double())#.to_sparse_csr()
          
        out_R = sum_row_mat @ spml.SparseCSRTensor(out_R).T#.to_sparse_csr().t()
   
        out_R = out_R.T
        
        # out_R = out_R.to_dense()/out_R.to_dense().sum(0)

        # out_R = out_R.to_sparse()
        # else:
            
        #     sum_row_mat = 1/torch.sparse.sum(out_R, dim = 0).coalesce().values()
        #     sum_row_mat = torch.sparse_coo_tensor([np.arange(sum_row_mat.shape[0]).tolist(),
        #                                               np.arange(sum_row_mat.shape[0]).tolist()], sum_row_mat, 
        #                                               (sum_row_mat.shape[0], sum_row_mat.shape[0])).double().to_sparse_csr()
              
        #     out_R = sum_row_mat @ out_R.to_sparse_csr().t()
    
        #     out_R = out_R.t().to_sparse_coo()

        
        # out_R = make_sparse_torch(grid.neigh_R0)
        
        
        
        row = np.array(grid.mask_edges)[:,0].tolist()
        col = np.array(grid.mask_edges)[:,1].tolist()
        
        edge_attr = self.edge_model(x[row], x[col])#, edge_attr.unsqueeze(1)) #+self.edge_model(x[col], x[row], edge_attr_i)
        # edge_attr = torch.zeros_like(edge_attr)
        # out =  edge_attr  #torch.nn.functional.relu(edge_attr) # torch.nn.functional.leaky_relu(edge_attr)
        sz = grid.gdata.x.shape[0]
        out = torch.sparse_coo_tensor([row, col], edge_attr.flatten(),(sz, sz)).double()#.to_dense()
        
        # if train:
        
        out = spml.SparseCSRTensor(out)
        
        # print(out.shape)
        # out0 = torch.zeros((out.shape[0], out.shape[0])).double()
        # print(out0.shape)
        
            
        return out, out_R# + torch.sparse_coo_tensor([grid.R0.nonzero()[0].tolist(), grid.R0.nonzero()[1].tolist()], 
                                                        #  torch.tensor(grid.R0.data).float(),
                                                        # (grid.R0.shape[0], grid.R0.shape[1])).double()

