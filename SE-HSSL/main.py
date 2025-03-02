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
from fairaug import orth_proj,balance_hyperedges

class MILNCELoss(nn.Module):
  
    def __init__(self,node_dim,edge_dim,d,tau,beta,batch_size,device,mean=True):
        super(MILNCELoss,self).__init__()
        
        self.d=d
        self.tau=tau 
        self.mean=mean
        self.batch_size=batch_size
        self.beta=torch.tensor(beta).to(device)
        self.disc=nn.Bilinear(node_dim,edge_dim,1).to(device) 
        self.device=device
    
    def f(self,x,tau):
        return torch.exp(x/tau)
    
    def Listwise_loss(self,n,e,hop_hyperedge,hop_hypernode):
        
        score_list=defaultdict(None) 
        for k in hop_hypernode.keys():
            score_list[k]=torch.zeros((hop_hypernode[k].shape[0],k)).to(self.device)
            
        losses=[]

        for k in hop_hypernode.keys():

            num_samples = len(hop_hypernode[k])
            num_batches = (num_samples - 1) // self.batch_size + 1
            indices = torch.arange(0, num_samples)

            for i in range(num_batches):
                
                ids = indices[i * self.batch_size: (i + 1) * self.batch_size] 
                node_idx=hop_hypernode[k][ids]
                anchor=n[node_idx,:] 
                
                anchor=torch.repeat_interleave(anchor.unsqueeze(1),self.d,dim=1)
                
                for j in range(k):
                    
                    contrast_edges_j=hop_hyperedge[k][ids,j]
                    contrast_obj=e[contrast_edges_j,:] # # [batch_size,sample_size,edge_dim]
                    hop_score=self.f(torch.sigmoid(self.disc(anchor,contrast_obj).squeeze()),self.tau).sum(dim=1) 
                    
                    score_list[k][ids,j]=hop_score
                        
        for k in hop_hypernode.keys():
            
            loss_k=torch.zeros(hop_hyperedge[k].shape[0]).to(self.device)
            for j in range(k-1):
                loss_k+=-torch.log(torch.min(score_list[k][:,j]/score_list[k][:,j:].sum(dim=1),self.beta))
            loss_k=loss_k/(k-1)
            losses.append(loss_k)
        
        return torch.cat(losses)
    
    def forward(self,n,e,hop_hyperedge,hop_hypernode):
        
        loss=self.Listwise_loss(n,e,hop_hyperedge,hop_hypernode)
        
        return loss.mean() if self.mean else loss.sum()    
    
def train(model,contrast_model,data,params,hop_hyperedge,hop_hypernode,optimizer,model_type,projection=False):
    
    features, hyperedge_index = data.features, data.hyperedge_index
    num_nodes, num_edges = data.num_nodes, data.num_edges

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Hypergraph Augmentation
    hyperedge_index1 = drop_incidence(hyperedge_index, params['drop_incidence_rate_1'])
    hyperedge_index2 = drop_incidence(hyperedge_index, params['drop_incidence_rate_2'])
    x1 = drop_features(features, params['drop_feature_rate_1'])
    x2 = drop_features(features, params['drop_feature_rate_2'])

    node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
    node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)
    node_mask = node_mask1 & node_mask2
    edge_mask = edge_mask1 & edge_mask2

    # Encoder
    n1, e1 = model(x1, hyperedge_index1, num_nodes, num_edges)
    n2, e2 = model(x2, hyperedge_index2, num_nodes, num_edges)
    n, e = model(features, hyperedge_index, num_nodes, num_edges)
    
    #n1, n2 = model.node_projection(n1), model.node_projection(n2)
    #e1, e2 = model.edge_projection(e1), model.edge_projection(e2)
    if projection:
        n, e = model.node_projection(n), model.edge_projection(e)
    
    if model_type in ['hssl','hssl_ng','hssl_n']:
        loss_n=cca_loss(n1,n2,num_nodes,params['lambda_n'],params['device'])
    else:
        loss_n=0
    
    if model_type in ['hssl','hssl_ng']:
        loss_g=cca_loss(e1,e2,num_edges,params['lambda_g'],params['device'])
    else:
        loss_g=0
    
    if model_type in ['hssl']:
        loss_m=contrast_model(n,e,hop_hyperedge,hop_hypernode)
    else:
        loss_m = 0
    
    loss = loss_n + params['w_g'] * loss_g + params['w_m'] * loss_m

    #print(loss_n.item(),loss_g.item(),loss_m.item())
    #print(loss_n.item(),loss_g.item())
    
    loss.backward()
    optimizer.step()
    
    return loss

# Fair variant
def FairHSSL_Training(model,contrast_model,data,params,hop_hyperedge,hop_hypernode,optimizer,model_type,projection=False):
    
    params=model.params

    features, hyperedge_index = data.x, data.hyperedge_index
    orth_features = orth_proj(features,data.sens_idx) 
    num_nodes, num_edges = data.num_nodes, data.num_hyperedges
    
    if hasattr(data,'sens_idx'):
        node_groups = features[:,data.sens_idx] 
        balanced_hyperedge_index = balance_hyperedges(hyperedge_index, node_groups, dname=args.dname) 
        
    for epoch in tqdm(range(1, params['n_epoch'] + 1)):
    
        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        if params['edge_aug'] == 'aos':
            hyperedge_index1=drop_incidence(hyperedge_index, params['drop_incidence_rate_1'])
            hyperedge_index2 = drop_incidence(balanced_hyperedge_index, params['drop_incidence_rate_2'])
        else:
            hyperedge_index1 = drop_incidence(hyperedge_index, params['drop_incidence_rate_1'])
            hyperedge_index2 = drop_incidence(hyperedge_index, params['drop_incidence_rate_2'])
        
        # Feature Augmentation
        if params['feat_aug'] == 'orth_proj':
            x1 =  drop_features(features, params['drop_feature_rate_1']) 
            x2 = drop_features(orth_features, params['drop_feature_rate_2'])
        else:
            x1 = drop_features(features, params['drop_feature_rate_1'])
            x2 = drop_features(features, params['drop_feature_rate_2'])

        node_mask1, edge_mask1 = valid_node_edge_mask(hyperedge_index1, num_nodes, num_edges)
        node_mask2, edge_mask2 = valid_node_edge_mask(hyperedge_index2, num_nodes, num_edges)

        # Encoder
        n1, e1 = model(x1, hyperedge_index1, num_nodes, num_edges)
        n2, e2 = model(x2, hyperedge_index2, num_nodes, num_edges)
        n, e = model(features, hyperedge_index, num_nodes, num_edges)
        
        if model_type in ['hssl','hssl_ng','hssl_n']:
            loss_n=cca_loss(n1,n2,num_nodes,params['lambda_n'],params['device'])
        else:
            loss_n=0
        
        if model_type in ['hssl','hssl_ng']:
            loss_g=cca_loss(e1,e2,num_edges,params['lambda_g'],params['device'])
        else:
            loss_g=0
        
        if model_type in ['hssl']:
            loss_m=contrast_model(n,e,hop_hyperedge,hop_hypernode)
        else:
            loss_m = 0
        
        loss = loss_n + params['w_g'] * loss_g + params['w_m'] * loss_m
        
        loss.backward()
        optimizer.step()

def generate_sample(data,params,save=True,seed=0):
    
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
            hop_hypernode[k+1]=ids # {2:[...],3:[...]}: ndarray
            
    for k in hop_hypernode.keys():
        hop_hyperedge[k]=torch.tensor([sample_list[i] for i in hop_hypernode[k]])
    
    if save:
        save_samples({'node':hop_hypernode,'edge':hop_hyperedge},data.name,params['K'])
    
    return hop_hypernode,hop_hyperedge

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('SE-HSSL unsupervised learning.')
    parser.add_argument('--dataset', type=str, default='cora', 
        choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 'dblp_coauthor', 
                    'zoo', '20newsW100', 'Mushroom', 'NTU2012', 'ModelNet40'])
    parser.add_argument('--model_type', type=str, default='hssl', choices=['hssl'])
    parser.add_argument('--is_fair',type=bool,default=False)
    parser.add_argument('--sample_seed', type=str, default=0, choices=['spcl'])
    parser.add_argument('--verbose_iter', type=int, default=50)
    parser.add_argument('--num_seeds', type=int, default=2)
    args = parser.parse_args(args=[])

    params = yaml.safe_load(open('config.yaml'))[args.dataset]
    
    data = DatasetLoader().load(args.dataset).to(params['device'])
    #sample_dict=load_samples(data.name,params['K'])
    #hop_hypernode,hop_hyperedge=sample_dict['node'],sample_dict['edge']
    #hop_hypernode,hop_hyperedge=generate_sample(data,params,seed=args.sample_seed)
    sample_dict=load_samples(args.dataset,params['K'])
    hop_hypernode,hop_hyperedge=sample_dict['node'],sample_dict['edge']
    
    accs = []

    for seed in range(args.num_seeds):  
        
        fix_seed(seed)
        
        encoder = HyperEncoder(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        model = TriCL(encoder, params['proj_dim']).to(params['device'])
        
        contrast_model=MILNCELoss(params['hid_dim'],params['hid_dim'],params['d'],params['tau'],params['beta'],params['batch_size'],params['device'],mean=False)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        if args.is_fair:
            FairHSSL_Training(model,contrast_model,data,params,hop_hyperedge,hop_hypernode,optimizer,
                              args.model_type,projection=False)
        else:
            for epoch in tqdm(range(1, params['n_epoch'] + 1)):
                loss = train(model,contrast_model,data,params,hop_hyperedge,hop_hypernode,optimizer,
                            args.model_type,projection=False)
            
        acc = node_classification_eval(model,data,lr=params['lr_lr'],max_epoch=params['lr_num_epochs'],
                                       lr_weight=params['lr_wd'])

        accs.append(acc)
        acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
        print(f'seed: {seed}, train_acc: {acc_mean[0]:.2f}+-{acc_std[0]:.2f}, '
            f'valid_acc: {acc_mean[1]:.2f}+-{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}+-{acc_std[2]:.2f}')

    accs = np.array(accs).reshape(-1, 3)
    accs_mean = list(np.mean(accs, axis=0))
    accs_std = list(np.std(accs, axis=0))
    print(f'[Final] dataset: {args.dataset}, test_acc: {accs_mean[2]:.2f}+-{accs_std[2]:.2f}')