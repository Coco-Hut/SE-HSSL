import random
from itertools import permutations
from collections import defaultdict
import numpy as np
import pickle
import torch
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_scatter import scatter_add

def save_samples(samples,data_name,k):

    with open('./samples/{}_{}.pkl'.format(data_name,k),'wb') as pkl_obj:
        pickle.dump(samples,pkl_obj)

def load_samples(data_name,k):

    with open('./samples/{}_{}.pkl'.format(data_name,k),'rb') as pkl_obj:
        data=pickle.load(pkl_obj)
    
    return data

def save_model(model,model_name,data_name):

    with open('./model_files/{}_{}.pkl'.format(model_name,data_name),'wb') as pkl_obj:
        pickle.dump(model,pkl_obj)
        
def load_model(model_name,data_name):

    with open('./model_files/{}_{}.pkl'.format(model_name,data_name),'rb') as pkl_obj:
        model=pickle.load(pkl_obj)
    
    return model

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def drop_features(x: Tensor, p: float):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def filter_incidence(row: Tensor, col: Tensor, hyperedge_attr: OptTensor, mask: Tensor):
    return row[mask], col[mask], None if hyperedge_attr is None else hyperedge_attr[mask]


def drop_incidence(hyperedge_index: Tensor, p: float = 0.2):
    if p == 0.0:
        return hyperedge_index
    
    row, col = hyperedge_index
    mask = torch.rand(row.size(0), device=hyperedge_index.device) >= p
    
    row, col, _ = filter_incidence(row, col, None, mask)
    hyperedge_index = torch.stack([row, col], dim=0)
    return hyperedge_index


def drop_nodes(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_nodes, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[drop_idx, :] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def drop_hyperedges(hyperedge_index: Tensor, num_nodes: int, num_edges: int, p: float):
    if p == 0.0:
        return hyperedge_index

    drop_mask = torch.rand(num_edges, device=hyperedge_index.device) < p
    drop_idx = drop_mask.nonzero(as_tuple=True)[0]

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    H[:, drop_idx] = 0
    hyperedge_index = H.to_sparse().indices()

    return hyperedge_index


def valid_node_edge_mask(hyperedge_index: Tensor, num_nodes: int, num_edges: int):
    ones = hyperedge_index.new_ones(hyperedge_index.shape[1])
    Dn = scatter_add(ones, hyperedge_index[0], dim=0, dim_size=num_nodes)
    De = scatter_add(ones, hyperedge_index[1], dim=0, dim_size=num_edges)
    node_mask = Dn != 0
    edge_mask = De != 0
    return node_mask, edge_mask


def common_node_edge_mask(hyperedge_indexs, num_nodes: int, num_edges: int):
    hyperedge_weight = hyperedge_indexs[0].new_ones(num_edges)
    node_mask = hyperedge_indexs[0].new_ones((num_nodes,)).to(torch.bool)
    edge_mask = hyperedge_indexs[0].new_ones((num_edges,)).to(torch.bool)

    for index in hyperedge_indexs:
        Dn = scatter_add(hyperedge_weight[index[1]], index[0], dim=0, dim_size=num_nodes)
        De = scatter_add(index.new_ones(index.shape[1]), index[1], dim=0, dim_size=num_edges)
        node_mask &= Dn != 0
        edge_mask &= De != 0
    return node_mask, edge_mask


def hyperedge_index_masking(hyperedge_index, num_nodes, num_edges, node_mask, edge_mask):
    if node_mask is None and edge_mask is None:
        return hyperedge_index

    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (num_nodes, num_edges)).to_dense()
    if node_mask is not None and edge_mask is not None:
        masked_hyperedge_index = H[node_mask][:, edge_mask].to_sparse().indices()
    elif node_mask is None and edge_mask is not None:
        masked_hyperedge_index = H[:, edge_mask].to_sparse().indices()
    elif node_mask is not None and edge_mask is None:
        masked_hyperedge_index = H[node_mask].to_sparse().indices()
    return masked_hyperedge_index

def clique_expansion(hyperedge_index):
    edge_set = set(hyperedge_index[1].tolist())
    adjacency_matrix = []
    for edge in edge_set:
        mask = hyperedge_index[1] == edge
        nodes = hyperedge_index[:, mask][0].tolist()
        for e in permutations(nodes, 2):
            adjacency_matrix.append(e)
    
    adjacency_matrix = list(set(adjacency_matrix))
    adjacency_matrix = torch.LongTensor(adjacency_matrix).T.contiguous()
    return adjacency_matrix.to(hyperedge_index.device)

def search_k_hop_edge(data,clique_index,hyperedge_index,device='cpu',K=1):
    
    if device=='cpu':
        clique_index=clique_index.cpu()
        hyperedge_index=hyperedge_index.cpu()
    
    H = torch.sparse_coo_tensor(hyperedge_index, \
        hyperedge_index.new_ones((hyperedge_index.shape[1],)), (data.num_nodes, data.num_edges)).to_dense().to(torch.float32)
    
    A = torch.sparse_coo_tensor(clique_index, \
        hyperedge_index.new_ones((clique_index.shape[1],)), (data.num_nodes, data.num_nodes)).to_dense().to(torch.float32)

    neighbor_list=defaultdict(list)
    
    M=A
    for k in range(K):
        if k==0:
            for i in range(H.shape[0]):
                neighbor_list[i].append(torch.where(H[i]>0)[0])
        else:
            for i in range(M.shape[0]):
                
                neighbor_ids=torch.where(M[i]>0)[0]
                
                k_hop_edge=torch.where(H[neighbor_ids].sum(dim=0))[0]
                
                less_k_edge=torch.cat(neighbor_list[i])
               
                k_hop_edge=k_hop_edge[torch.isin(k_hop_edge,less_k_edge,invert=True)]
                
                neighbor_list[i].append(k_hop_edge)
               
                M=torch.mm(M,A)
          
    for v_i in range(len(neighbor_list)):
        
        while len(neighbor_list[v_i][-1])==0:
            neighbor_list[v_i].pop()
        
        k_plus_one_hop=torch.arange(H.shape[1]).to(device)
        less_k_edge=torch.cat(neighbor_list[v_i])
        k_plus_one_hop=k_plus_one_hop[torch.isin(k_plus_one_hop,less_k_edge,invert=True)]
        
        if len(k_plus_one_hop)!=0:
            neighbor_list[v_i].append(k_plus_one_hop)
            
    return neighbor_list