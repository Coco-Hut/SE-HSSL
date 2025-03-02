import pickle
import torch
import math
import os

def orth_proj(x,sens_idx):
    
    def unit(v):
        norm_0 = torch.norm(v, p=2)
        unit_v = v / (norm_0 + epsilon)
        return unit_v
    
    groups = x[:,sens_idx]
    
    idx_zero=torch.where(groups==0)[0]
    idx_one=torch.where(groups==1)[0]
    
    x_0=x[idx_zero]
    x_1=x[idx_one]
    
    mu_0=torch.mean(x_0,dim=0)
    mu_1=torch.mean(x_1,dim=0)
    
    epsilon = 1e-8
    v_0 = torch.sum(x_0,dim=0)
    v_1 = torch.sum(x_1,dim=0)

    unit_v0=unit(v_0)
    unit_v1=unit(v_1)
    
    bias_v=unit_v0-unit_v1
    unit_bias=unit(bias_v)
    
    ip_bias=(x @ unit_bias)
    debias_x = x-ip_bias.unsqueeze(1) @ unit_bias.unsqueeze(0)
    
    return debias_x

def balance_hyperedges(hyperedge_index, node_groups, beta=1, 
                       dname=None, load_exist=True, load_url='lib_samples',verbose=True):
    
    file_path = './{}/{}_{}_{}.pkl'.format(load_url,dname,'balance_edge',beta)

    if load_exist and os.path.exists(file_path):
        with open(file_path,'rb') as pkl_obj:
            balanced_hyperedge_index=pickle.load(pkl_obj)
    else:
        edge_dict = {}  
        for node, edge in hyperedge_index.T:
            edge_dict.setdefault(edge.item(), []).append(node.item()) 

        node_group_0 = [i for i, x in enumerate(node_groups) if x == 0]
        node_group_1 = [i for i, x in enumerate(node_groups) if x == 1]
        
        new_node_list = []
        new_edge_list = []

        k=0
        
        for edge, nodes in edge_dict.items():
            
            group_0 = [n for n in nodes if node_groups[n] == 0] 
            group_1 = [n for n in nodes if node_groups[n] == 1] 

            len_0, len_1 = len(group_0), len(group_1)
            
            num_sample=int(math.ceil(beta * abs(len_1 - len_0))) 
            
            if len_0 < len_1 and len_0 > 0:
                group_0 += [group_0[i] for i in torch.randint(0, len_0, (num_sample,)).tolist()]
            elif len_0 < len_1 and len_0 == 0:
                aug_idx = torch.randint(0, len(node_group_0), (num_sample,)).tolist()
                group_0 = [node_group_0[i] for i in aug_idx]  
            elif len_1 < len_0 and len_1 > 0:
                group_1 += [group_1[i] for i in torch.randint(0, len_1, (num_sample,)).tolist()]
            elif len_1 < len_0 and len_1 == 0:
                aug_idx = torch.randint(0, len(node_group_1), (num_sample,)).tolist()
                group_1 = [node_group_1[i] for i in aug_idx]  

            balanced_nodes = group_0 + group_1
            new_node_list.extend(balanced_nodes)
            new_edge_list.extend([edge] * len(balanced_nodes))
            
            k+=1
            
            if verbose:
                if k%1000==0:
                    print('Traverse {}'.format(k))

        balanced_hyperedge_index = torch.tensor([new_node_list, new_edge_list], dtype=torch.long)
        
        with open(file_path,'wb') as pkl_obj:
            pickle.dump(balanced_hyperedge_index,pkl_obj)
    
    return balanced_hyperedge_index.to(hyperedge_index.device)