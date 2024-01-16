import torch
import numpy as np
import torch.nn.functional as F

def cca_loss(z1,z2,N,lbd,device):
    
    z1 = (z1 - z1.mean(0)) / z1.std(0)
    z2 = (z2 - z2.mean(0)) / z2.std(0)
    
    c = torch.mm(z1.T, z2)
    c1 = torch.mm(z1.T, z1)
    c2 = torch.mm(z2.T, z2)

    c = c / N
    c1 = c1 / N
    c2 = c2 / N

    loss_inv = -torch.diagonal(c).sum()
    iden = torch.tensor(np.eye(c.shape[0])).to(device)
    loss_dec1 = (iden - c1).pow(2).sum()
    loss_dec2 = (iden - c2).pow(2).sum()
    
    return loss_inv + lbd * (loss_dec1 + loss_dec2)

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss, mode, intraview_negs=False):
        super(DualBranchContrast, self).__init__()
        self.loss = loss

    def forward(self, h1=None, h2=None):
        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)
        return (l1 + l2) * 0.5

class InfoNCE(object):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

    def __call__(self, anchor, sample) -> torch.FloatTensor:
        loss = self.compute(anchor, sample)
        return loss