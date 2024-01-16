import argparse
import random
import yaml
from tqdm import tqdm
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from collections import defaultdict
from loader import DatasetLoader
from contrast_loss import cca_loss
from tricl_encoder import HyperEncoder, TriCL
from utils import fix_seed,drop_features, drop_incidence, valid_node_edge_mask, hyperedge_index_masking
from utils import search_k_hop_edge,clique_expansion,save_samples,load_samples
from evaluation import node_classification_eval


def generate_sample(data,params,name,save=True,seed=0):
    
    # clique expansion: node-node adjacency matrix
    clique_index=clique_expansion(data.hyperedge_index)
    hyperedge_index=data.hyperedge_index
    
    neighbor_list=search_k_hop_edge(data,clique_index,hyperedge_index,'cpu',K=params['K'])

    np.random.seed(seed)

    sample_list=defaultdict(list)
    for v_i in range(len(neighbor_list)):
        for k in range(len(neighbor_list[v_i])):
            sample_list[v_i].append(np.random.choice(neighbor_list[v_i][k].numpy(),params['d']).tolist())
    
    hop_hypernode=dict()
    hop_hyperedge=dict()

    k_hop_n=np.array(list(map(len,sample_list.values())))

    for k in range(params['K']+1):
        ids=np.where(k_hop_n==(k+1))[0]
        if len(ids):
            hop_hypernode[k+1]=ids 
            
    for k in hop_hypernode.keys():
        hop_hyperedge[k]=torch.tensor([sample_list[i] for i in hop_hypernode[k]])
    
    if save:
        save_samples({'node':hop_hypernode,'edge':hop_hyperedge},name,params['K'])
    
    return hop_hypernode,hop_hyperedge

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('TriCL unsupervised learning.')
    parser.add_argument('--dataset', type=str, default='pubmed', 
        choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 'dblp_coauthor', 
                    'zoo', '20newsW100', 'Mushroom', 'NTU2012', 'ModelNet40'])
    parser.add_argument('--model_type', type=str, default='hssl', choices=['hssl'])
    parser.add_argument('--sample_seed', type=str, default=0, choices=['hssl'])
    parser.add_argument('--verbose_iter', type=int, default=50)
    parser.add_argument('--num_seeds', type=int, default=5)
    args = parser.parse_args(args=[])

    params = yaml.safe_load(open('config.yaml'))[args.dataset]
    
    data = DatasetLoader().load(args.dataset).to(params['device'])
    #sample_dict=load_samples(data.name,params['K'])
    #hop_hypernode,hop_hyperedge=sample_dict['node'],sample_dict['edge']
    hop_hypernode,hop_hyperedge=generate_sample(data,params,name=args.dataset,seed=args.sample_seed)