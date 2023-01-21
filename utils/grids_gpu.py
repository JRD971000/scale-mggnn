#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:30:04 2023

@author: alitaghibakhshi
"""

import networkx as nx
import torch
import torch_geometric as tg
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import torch_geometric
import fem
from Unstructured import rand_grid_gen, from_scipy_sparse_matrix, from_networkx, lloyd_aggregation
import pyamg
import scipy

def graph_from_matrix(A, agg_op):
    n = A.shape[0]

    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)
    x = torch.ones(n) / n

    # Create cluster feature
    clusters = np.array(agg_op.argmax(axis=1)).flatten()
    cluster_adj = {} # 0 if in same cluster, 1 if not
    for (u, v) in nx.edges(G):
        adj = (0 if (clusters[u] == clusters[v]) else 1)
        cluster_adj[(u, v)] = adj

    nx.set_edge_attributes(G, cluster_adj, 'cluster_adj')
    nx_data = tg.utils.from_networkx(G, None, ['weight', 'cluster_adj'])
    return tg.data.Data(x=x, edge_index=nx_data.edge_index, edge_attr=abs(nx_data.edge_attr.float()))

def graph_from_matrix_basic(A):
    n = A.shape[0]

    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)
    x = torch.ones(n) / n

    # Create cluster feature
    cluster_adj = {} # 0 if in same cluster, 1 if not
    for (u, v) in nx.edges(G):
        cluster_adj[(u, v)] = 1.0 / n

    nx.set_edge_attributes(G, cluster_adj, 'cluster_adj')
    nx_data = tg.utils.from_networkx(G, None, ['weight', 'cluster_adj'])
    return tg.data.Data(x=x, edge_index=nx_data.edge_index, edge_attr=abs(nx_data.edge_attr.float()))

class MyMesh:
    def __init__(self, mesh):
        
        self.nv = mesh.points[:,0:2].shape[0]
        self.X = mesh.points[:,0:1].flatten() * ((self.nv/50)**0.5)
        self.Y = mesh.points[:,1:2].flatten() * ((self.nv/50)**0.5)

        self.E = mesh.cells[1].data
        self.V = mesh.points[:,0:2]
        
        self.ne = len(mesh.cells[1].data)
        
        e01 = self.E[:,[0,1]]
        e02 = self.E[:,[0,2]]
        e12 = self.E[:,[1,2]]
    
        e01 = tuple(map(tuple, e01))
        e02 = tuple(map(tuple, e02))
        e12 = tuple(map(tuple, e12))
        
        e = list(set(e01).union(set(e02)).union(set(e12)))
        self.N = [i for i in range(self.X.shape[0])]
        self.Edges = e
        self.num_edges = len(e)
        
class Old_Grid(object):
    
    def __init__(self, A, mesh):

        self.A = A.tocsr()

        self.num_nodes = mesh.nv
        #self.edges = set_edge
        self.mesh = mesh
        
        active = np.ones(self.num_nodes)
        self.active = active
        
        self.G = nx.from_scipy_sparse_matrix(self.A, edge_attribute='weight', parallel_edges=False)

  
        self.x = torch.cat((torch.from_numpy(self.active).unsqueeze(1), \
                        torch.from_numpy(self.active).unsqueeze(1)),dim=1).float()

        
        edge_index, edge_attr = from_scipy_sparse_matrix(abs(self.A))
        edge_index4P, edge_attr4P = from_scipy_sparse_matrix(self.A)
        
        list_neighbours1 = []
        list_neighbours2 = []
        for node in range(self.num_nodes):
            a =  list(self.G.edges(node,data = True))
            l1 = []
            l2 = []
            for i in range(len(a)):
                l1.append(a[i][1])
                l2.append(abs(np.array(list(a[i][-1].values())))[0])
                
            list_neighbours1.append(l1)
            list_neighbours2.append(l2)
                
        self.list_neighbours = [list_neighbours1, list_neighbours2]
        
        self.data = Data(x=self.x, edge_index=edge_index, edge_attr= edge_attr.float())
        self.data4P = Data(x=self.x, edge_index=edge_index4P, edge_attr= edge_attr4P.float())

        
    def subgrid(self, node_list):

        sub_x = self.x[node_list]
        sub_data = from_networkx(self.G.subgraph(node_list))
        sub_data = Data(x=sub_x, edge_index=sub_data.edge_index, edge_attr= abs(sub_data.weight.float()))
        
        return sub_data
    
        
    def node_hop_neigh(self, node, K):
        
        return list(nx.single_source_shortest_path(self.G, node, cutoff=K).keys())
    
    def aggop_gen(self, ratio):
        
        elem_adj = np.zeros((len(self.mesh.E.tolist()), len(self.mesh.E.tolist())))

        for i, e1 in enumerate(self.mesh.E.tolist()):
            for j, e2 in enumerate(self.mesh.E.tolist()):
                if i!= j:
                    if len(set(e1) - set(e2)) == 1:
                        elem_adj[i,j] = 1
                        
        elem_agg = lloyd_aggregation(scipy.sparse.csr_matrix(elem_adj), ratio)
        node_agg = np.zeros((self.mesh.V.shape[0], elem_agg[1].shape[0]))

        for i, e in enumerate(self.mesh.E.tolist()):
            for node in e:
                node_agg[node, elem_agg[-1][i]] = 1
                
        elem_dict = []
        for e in self.mesh.E.tolist():
            elem_dict.append(set(e))
            

        self.aggop = (scipy.sparse.csr_matrix(node_agg), 0, elem_dict, elem_agg[-1], 0)


        all_eye = np.eye(self.aggop[0].shape[0])
        
        self.R = {}
        for i in range(self.aggop[0].shape[1]):
            self.R[i] = all_eye[self.aggop[0].transpose()[i].nonzero()[-1].tolist(), :]
        


        list_w = []
        for i in range(self.aggop[0].shape[0]):
            w = self.aggop[0][i].indices.shape[0]
            if w>1:
                list_w.append(1/(w-1))
            else:
                list_w.append(1)
        vec_w = np.array(list_w)

        weighted_eye = all_eye * vec_w

        self.R_tilde = {}
        for i in range(self.aggop[0].shape[1]):
            self.R_tilde[i] = weighted_eye[self.aggop[0].transpose()[i].nonzero()[-1].tolist(), :]
                        
        return 
    
        
def structured_2d_poisson_dirichlet(n_pts_x, n_pts_y,
                                        xdim=(0,1), ydim=(0,1),
                                        epsilon=1.0, theta=0.0):
        '''
        Creates a 2D poisson system on a structured grid, discretized using finite elements.
        Dirichlet boundary conditions are assumed.
        Parameters
        ----------
        n_pts_x : integer
          Number of inner points in the x dimension (not including boundary points)
        n_pts_y : integer
          Number of inner points in the y dimension (not including boundary points)
        xdim : tuple (float, float)
          Bounds for domain in x dimension.  Represents smallest and largest x values.
        ydim : tuple (float, float)
          Bounds for domain in y dimension.  Represents smallest and largest y values.
        Returns
        -------
        Grid object with given parameters.
        '''

        x_pts = np.linspace(xdim[0], xdim[1], n_pts_x+2)[1:-1]
        y_pts = np.linspace(xdim[0], ydim[1], n_pts_y+2)[1:-1]
        delta_x = abs(x_pts[1] - x_pts[0])
        delta_y = abs(y_pts[1] - y_pts[0])

        xx, yy = np.meshgrid(x_pts, y_pts)
        xx = xx.flatten()
        yy = yy.flatten()

        grid_x = np.column_stack((xx, yy))
        n = n_pts_x * n_pts_y
        A = sp.lil_matrix((n, n), dtype=np.float64)

        stencil = pyamg.gallery.diffusion_stencil_2d(epsilon=epsilon, theta=theta, type='FD')
        print(stencil)

        for i in range(n_pts_x):
            for j in range(n_pts_y):
                idx = i + j*n_pts_x

                A[idx, idx] = stencil[1,1]
                has_left = (i>0)
                has_right = (i<n_pts_x-1)
                has_down = (j>0)
                has_up = (j<n_pts_y-1)

                # NSEW connections
                if has_up:
                    A[idx, idx + n_pts_x] = stencil[0, 1]
                if has_down:
                    A[idx, idx - n_pts_x] = stencil[2, 1]
                if has_left:
                    A[idx, idx - 1] = stencil[1, 0]
                if has_right:
                    A[idx, idx + 1] = stencil[1, 2]

                # diagonal connections
                if has_up and has_left:
                    A[idx, idx + n_pts_x - 1] = stencil[0, 0]
                if has_up and has_right:
                    A[idx, idx + n_pts_x + 1] = stencil[0, 2]
                if has_down and has_left:
                    A[idx, idx - n_pts_x - 1] = stencil[2, 0]
                if has_down and has_right:
                    A[idx, idx - n_pts_x + 1] = stencil[2, 2]
        A = A.tocsr()

        return A #Grid(A, grid_x)
    
    
def uns_grid(meshsz):
    
    old_g  = rand_grid_gen(meshsz, 'Poisson')
    
    return old_g

import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(a1, a2, a3):
    
  v1 = a1-a2
  v2 = a3-a2
  out = max(dotproduct(v1, v2) / (length(v1) * length(v2)), -1)
  out = min(out, 1)
  return math.acos(out)

def find_boundary(A, V):
    
    p = 0
    boundary = []
    while True:
        list_angle = []
        list_neighs = []
        neighs = A[p].nonzero()[-1].tolist()
        neighs.remove(p)
        sz = len(neighs)
        for i in range(1,sz):
            for j in range(i):
                ang = angle(V[neighs[i]], V[p], V[neighs[j]])
                list_angle.append(ang)
                list_neighs.append((neighs[i], neighs[j]))
                
        idx_max = np.argmax(np.array(list_angle))
        p1, p2 = list_neighs[idx_max][0], list_neighs[idx_max][1]
        
        if p1 not in boundary:
            p = p1
            boundary.append(p)
        elif p2 not in boundary:
            p = p2
            boundary.append(p)
        else:
            break
    return boundary
        
        
    
    
def refine_grid(grid, levels, ratio = None):
    
    msh = fem.mesh(grid.mesh.V, grid.mesh.E)
    msh.refine(levels)
    A, _ = fem.gradgradform(msh, PDE = 'Helmholtz')
    A = scipy.sparse.csr_matrix(A)
    boundary = find_boundary(A, msh.V)
    if ratio is None:
        ratio = 25*((A.shape[0]/600)**0.5)/A.shape[0]
    new_grid =  Grid_PWA(A, msh, ratio, hops = grid.hops, 
                          cut=grid.cut, h = 1, nu = 0, BC = grid.BC, boundary= boundary) 
    
    return new_grid
    

        
class Grid_PWA():
    def __init__(self, A, mesh, ratio, hops = 1, cut=1, h = 1, nu = 0, BC = 'Neumann', boundary = None):
        '''
        Initializes the grid object
        Parameters
        ----------
        A_csr : scipy.sparse.csr_matrix
          CSR matrix representing the underlying PDE
        x : numpy.ndarray
          Positions of the points of each node.  Should have shape (n_pts, n_dim).
        '''

        self.A = A
        self.BC = BC
            
        self.mesh = mesh
        self.x = self.mesh.V
        
        # if BC == 'Dirichlet':
        #     self.apply_bc(1e-8, boundary)
            
            
        self.hops = hops
        self.cut = cut
        self.ratio = ratio
        self.dict_nodes_neighbors_cut = {}
        self.dict_nodes_neighbors_hop = {}
        self.h = h
        self.nu = nu
        
        modif = scipy.sparse.diags([self.nu * (self.h ** 2) for _ in range(self.A.shape[0])])
        self.A = (1/(self.h ** 2)) * (self.A + modif)
        
        A_cut = scipy.sparse.csr_matrix(scipy.sparse.identity(self.A.shape[0]))
        for _ in range(self.cut):
            
            A_cut = A_cut @ self.A
            
        for n in range(self.A.shape[0]):
            self.dict_nodes_neighbors_cut[n] = set(A_cut[n].nonzero()[-1].tolist())
            
            
        self.dict_nodes_neighbors_hop = {}
        
        if hops != -1:
            A_hop = scipy.sparse.csr_matrix(scipy.sparse.identity(self.A.shape[0]))
            for _ in range(self.hops):
                
                A_hop = A_hop @ self.A
                
            for n in range(self.A.shape[0]):
                self.dict_nodes_neighbors_hop[n] = set(A_hop[n].nonzero()[-1].tolist())
            
        self.aggop_gen(self.ratio, self.cut, boundary=boundary)

        
    
    def to(self, device):
        
        self.gdata = self.gdata.to(device)
        self.gmask = self.gmask.to(device)
        
    @property

    def networkx(self):
        return nx.from_scipy_sparse_matrix(self.A, edge_attribute='weight', parallel_edges=False, create_using=nx.DiGraph)

 
    def apply_bc(self, zer):
                    
        for n in self.boundary:
            nzs = self.A[n].nonzero()[-1].tolist()
            for m in nzs:
                self.A[n,m] = zer
                self.A[m,n] = zer
            self.A[n,n] = 1.0
        
    
    def global_Lap_eig(self):
        
        sz = self.aggop[0].shape[0]
        
        D = sp.diags((np.array(self.networkx.degree)[:,1]/2)**(-0.5))
        L = sp.eye(sz) - D @ self.A @ D
        
        masks = self.gmask
        num_eigs = 20
        evals, evecs = sp.linalg.eigsh(L, k=num_eigs)
        
        x = torch.zeros(sz, 2*num_eigs+1).float()
        x[self.boarder_hops, 0] = 1.0
        x[:, [i for i in range(1,num_eigs+1)]]  = torch.tensor(evals).float()
        x[:, [i for i in range(num_eigs+1,2*num_eigs+1)]] = torch.tensor(evecs).float()
        
        edge_index, e_w0 = from_scipy_sparse_matrix(self.A)
        e_w1 = torch.tensor([masks[edge_index[0, i], edge_index[1, i]] for i in range(edge_index[0].shape[0])])
        
        edge_attr = torch.cat((e_w0.unsqueeze(1), e_w1.unsqueeze(1)), dim = 1)
        
        data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr.float())
        
        self.gdata = data
        
        
    def data(self):
        
        sz = self.aggop[0].shape[0]
        masks = self.gmask

        x = torch.zeros(sz)
        x[self.boarder_hops] = 1.0
        edge_index, e_w0 = from_scipy_sparse_matrix(self.A)
        e_w1 = torch.tensor([masks[edge_index[0, i], edge_index[1, i]] for i in range(edge_index[0].shape[0])])
        
        edge_attr = torch.cat((e_w0.unsqueeze(1), e_w1.unsqueeze(1)), dim = 1)
        
        data = Data(x = x.unsqueeze(1).float(), edge_index = edge_index, edge_attr = edge_attr.float())
        
        return data
    
    def aggop_gen(self, ratio, cut, node_agg=None, boundary = None):
        
        if node_agg is None:
            
            self.aggop = lloyd_aggregation(self.A, ratio)  
        else:
            
            self.aggop = node_agg
        
        num_aggs = self.aggop[0].shape[1]
        list_aggs = {}
        list_cut_aggs = {}
        for i in range(num_aggs):
            list_aggs[i] = []
            list_cut_aggs[i] = set([])

        for i, n in enumerate(self.aggop[-1]):
            list_aggs[n].append(i)
            list_cut_aggs[n] = list_cut_aggs[n].union(self.dict_nodes_neighbors_cut[i])

        for i in range(num_aggs):
            list_cut_aggs[i] = list(list_cut_aggs[i])

        learn_nodes = {}
        boarder_hops = []
        for i in range(num_aggs):
            learn_nodes[i] = list(set(list_cut_aggs[i]) - set(list_aggs[i]))
            boarder_hops.extend(learn_nodes[i])

        # list_learn_edges = {}

        # for i in range(num_aggs):
        #     list_learn_edges[i] = list(self.networkx.subgraph(learn_nodes[i]).edges())

        mask_edges = []
        
        
        if self.hops != -1:
            for i in range(num_aggs):
                for node in learn_nodes[i]:
                    for j in list(self.dict_nodes_neighbors_hop[node]):
                        if j in learn_nodes[i]:
                            mask_edges.append((node, j))
                            
        else:
            for i in range(num_aggs):
                for node in learn_nodes[i]:
                    for j in learn_nodes[i]:
                        mask_edges.append((node, j))
                            
        mask_edges = list(set(mask_edges))

        sz = self.aggop[0].shape[0]
        
        mask_mat = scipy.sparse.csr_matrix((np.ones(len(mask_edges)), (np.array(mask_edges)[:,0].tolist(), np.array(mask_edges)[:,1].tolist())), shape=(sz, sz))


        # all_eye = np.eye(sz)
                
        R = {}
        R_hop = {}
        modified_R = {}
        
        for i in range(num_aggs):
            # R[i] = scipy.sparse.csr_matrix(all_eye[list_aggs[i], :])
            R_nodes = len(list_aggs[i])
            R[i] = scipy.sparse.csr_matrix((np.ones(R_nodes), (np.arange(R_nodes).tolist(), list_aggs[i])), shape=(R_nodes, sz))
            
            # R_hop[i] = scipy.sparse.csr_matrix(all_eye[list_cut_aggs[i], :])
            R_hop_nodes = len(list_cut_aggs[i])
            R_hop[i] = scipy.sparse.csr_matrix((np.ones(R_hop_nodes), (np.arange(R_hop_nodes).tolist(), list_cut_aggs[i])), shape=(R_hop_nodes, sz))

            list_ixs = []
            for e in list_aggs[i]:
                list_ixs.append(list_cut_aggs[i].index(e))
        
            modified_R[i] = scipy.sparse.csr_matrix((np.ones(R_nodes), (list_ixs, np.array(list_cut_aggs[i])[list_ixs])), shape=(R_hop_nodes, sz))
        
        
        l0 = []
        l1 = []
        for i in range(num_aggs):
            
            non_zeros = R_hop[i].nonzero()[-1].tolist()
            l0.extend(non_zeros)
            l1.extend([i for j in range(len(non_zeros))]) 
            
        R0 = scipy.sparse.csr_matrix((np.ones(len(l0)), (l0, l1)), shape=(sz, num_aggs))
        R0 = scipy.sparse.diags(1/R0.sum(axis=1).A.ravel()) @ R0
        R0 = R0.transpose()
        
        
        
        A_c = self.aggop[0].transpose() @ self.A @ self.aggop[0]
        l0 = []
        l1 = []
        for i in range(num_aggs):

            non_zeros = R_hop[i].nonzero()[-1].tolist()
            for neigh in A_c[i].nonzero()[-1].tolist():
                l0.extend(non_zeros)
                l1.extend([neigh for j in range(len(non_zeros))]) 

        neigh_R0 = scipy.sparse.csr_matrix((np.ones(len(l0)), (l0, l1)), shape=(sz, num_aggs))
        neigh_R0 = scipy.sparse.diags(1/neigh_R0.sum(axis=1).A.ravel()) @ neigh_R0
        neigh_R0 = neigh_R0.transpose()
        
            
        self.list_aggs = list_aggs
        self.list_cut_aggs = list_cut_aggs
        self.learn_nodes = learn_nodes
        # self.list_learn_edges = list_learn_edges
        self.mask_edges = mask_edges
        self.gmask = mask_mat
        self.boarder_hops = boarder_hops
        self.R = R
        self.R_hop = R_hop
        self.modified_R = modified_R
        self.R0 = R0
        self.neigh_R0 = neigh_R0
        self.gdata = self.data()

        if boundary is None:
            if self.mesh.E.shape[-1] == 3:
                max_b = self.A[0].nonzero()[-1][2]
                boundary = [i for i in range(1+max_b)]
                
            if self.mesh.E.shape[-1] == 4:
                
                boundary = []
                n_col = int(self.mesh.X.max()/0.04 + 1)
                n_row = int(len(self.mesh.X)/n_col)
                
                boundary.extend([i for i in range(n_col)])
                boundary.extend([i*n_col for i in range(n_row)])
                boundary.extend([(i+1)*n_col-1 for i in range(n_row)])
                boundary.extend([n_col*n_row - 1 - i for i in range(n_col)])

        self.boundary = boundary
        
        if self.BC == 'Dirichlet':
            self.apply_bc(1e-16)


        return 
    
    
    
    
    