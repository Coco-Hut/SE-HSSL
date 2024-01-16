from typing import Optional
import os.path as osp
import pickle

import torch
from torch_scatter import scatter_add
from torch.utils.data import random_split


class BaseDataset(object):
    def __init__(self, type: str, name: str, device: str = 'cpu'):
        self.type = type
        self.name = name
        self.device = device
        if self.type in ['cocitation', 'coauthorship']:
            self.dataset_dir = osp.join('../dataset', self.type, self.name) 
        else:
            self.dataset_dir = osp.join('../dataset', self.name)
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        self.load_dataset()
        self.preprocess_dataset()

    def load_dataset(self):
        
        '''
        1. features: node features matrix: [N x N]
        2. hypergraph: Hyperedge group dictionary: {Hyperedge_index:[n_1,n_2,...]}
        3. labels: node label tensor: [3,0,2,...]
        '''
        
        with open(osp.join(self.dataset_dir, 'features.pickle'), 'rb') as f:
            self.features = pickle.load(f)
        with open(osp.join(self.dataset_dir, 'hypergraph.pickle'), 'rb') as f:
            self.hypergraph = pickle.load(f)
        with open(osp.join(self.dataset_dir, 'labels.pickle'), 'rb') as f:
            self.labels = pickle.load(f)

    def load_splits(self, seed: int):
        with open(osp.join(self.split_dir, f'{seed}.pickle'), 'rb') as f:
            splits = pickle.load(f)
        return splits
        
    def preprocess_dataset(self):
        edge_set = set(self.hypergraph.keys())
        edge_to_num = {}
        num_to_edge = {}
        num = 0
        # 建立超边和index_num的对应
        for edge in edge_set:
            edge_to_num[edge] = num
            num_to_edge[num] = edge
            num += 1

        incidence_matrix = []
        processed_hypergraph = {}
        for edge in edge_set:
            nodes = self.hypergraph[edge]
            processed_hypergraph[edge_to_num[edge]] = nodes
            for node in nodes:
                incidence_matrix.append([node, edge_to_num[edge]])

        self.processed_hypergraph = processed_hypergraph
        self.features = torch.as_tensor(self.features.toarray())
        '''
        # [[node_list],
           [hyperedge_list]]
        '''
        self.hyperedge_index = torch.LongTensor(incidence_matrix).T.contiguous() 
        self.labels = torch.LongTensor(self.labels)
        self.num_nodes = int(self.hyperedge_index[0].max()) + 1
        self.num_edges = int(self.hyperedge_index[1].max()) + 1
        self.edge_to_num = edge_to_num
        self.num_to_edge = num_to_edge

        weight = torch.ones(self.num_edges)
        self.Dn = scatter_add(weight[self.hyperedge_index[1]], self.hyperedge_index[0], dim=0, dim_size=self.num_nodes)
        self.De = scatter_add(torch.ones(self.hyperedge_index.shape[1]), self.hyperedge_index[1], dim=0, dim_size=self.num_edges)

        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self

    def generate_random_split(self, train_ratio: float = 0.1, val_ratio: float = 0.1,
                              seed: Optional[int] = None, use_stored_split: bool = True):
        if use_stored_split:
            splits = self.load_splits(seed)
            train_mask = torch.tensor(splits['train_mask'], dtype=torch.bool, device=self.device)
            val_mask = torch.tensor(splits['val_mask'], dtype=torch.bool, device=self.device)
            test_mask = torch.tensor(splits['test_mask'], dtype=torch.bool, device=self.device)

        else:
            num_train = int(self.num_nodes * train_ratio)
            num_val = int(self.num_nodes * val_ratio)
            num_test = self.num_nodes - (num_train + num_val)

            if seed is not None:
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = torch.default_generator

            train_set, val_set, test_set = random_split(
                torch.arange(0, self.num_nodes), (num_train, num_val, num_test), 
                generator=generator)
            train_idx, val_idx, test_idx = \
                train_set.indices, val_set.indices, test_set.indices
            train_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            val_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)
            test_mask = torch.zeros((self.num_nodes,), device=self.device).to(torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

        '''
        [tensor([False, False, False,  ..., False, False, False], device='cuda:0'),
        tensor([False, False,  True,  ..., False,  True,  True], device='cuda:0'),
        tensor([ True,  True, False,  ...,  True, False, False], device='cuda:0')]
        '''
        
        return [train_mask, val_mask, test_mask]



class CoraCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'cora', **kwargs)


class CiteseerCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'citeseer', **kwargs)


class PubmedCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cocitation', 'pubmed', **kwargs)


class CoraCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('coauthorship', 'cora', **kwargs)


class DBLPCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('coauthorship', 'dblp', **kwargs)


class ZooDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'zoo', **kwargs)


class NewsDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', '20newsW100', **kwargs)


class MushroomDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('etc', 'Mushroom', **kwargs)


class NTU2012Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'NTU2012', **kwargs)


class ModelNet40Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__('cv', 'ModelNet40', **kwargs)