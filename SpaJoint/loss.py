import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss


def cor(m):
    m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def reduction_loss(embedding, identity_matrix, size):
    loss = torch.mean(torch.abs(torch.triu(cor(embedding), diagonal=1)))
    loss = loss + 1 / torch.mean(
        torch.abs(embedding - torch.mean(embedding, dim=0).view(1, size).repeat(embedding.size()[0], 1)))
    loss = loss + torch.mean(torch.abs(embedding))
    return loss


def cosine_sim(x, y):
    x = x / torch.norm(x, dim=1, keepdim=True)
    y = y / torch.norm(y, dim=1, keepdim=True)
    sim = torch.matmul(x, torch.transpose(y, 0, 1))

    return sim 

class EncodingLoss(nn.Module):
    def __init__(self, dim=64, use_gpu = True):
        super(EncodingLoss, self).__init__()
        if use_gpu:
            self.identity_matrix = torch.tensor(np.identity(dim)).float().cuda()
        else:
            self.identity_matrix = torch.tensor(np.identity(dim)).float()
        self.dim = dim
        
    def forward(self, st_embeddings, rna_embeddings, inf_neigh):
        # rna
        rna_embedding_cat = rna_embeddings[0]
        rna_reduction_loss = reduction_loss(rna_embeddings[0], self.identity_matrix, self.dim)
        
        # st
        st_reduction_loss = reduction_loss(st_embeddings[0][0:256], self.identity_matrix, self.dim)
        
        # cosine similarity loss    
        top_k_sim = torch.topk(
            torch.max(cosine_sim(st_embeddings[0][0:256,:], rna_embedding_cat), dim=1)[0],
            int(st_embeddings[0][0:256,:].shape[0]))
        sim_loss = torch.mean(top_k_sim[0])
        
        #knn cosine loss
        mat=cosine_sim(st_embeddings[0],st_embeddings[0])
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        # positive pairs
        p = torch.exp(mat)
        p = torch.mul(p, inf_neigh).sum(axis=1)
        
        ave = torch.div(p, k)
        loc_loss = - torch.log(ave).mean()
        a=0.3
        b=0.2
        loss = a*rna_reduction_loss + a*st_reduction_loss -b*sim_loss + (1-2*a-b)*loc_loss 
        print("rna_reduction_loss = {},st_reduction_loss={},sim_loss={},loc_loss={}."
              .format(rna_reduction_loss.item(),st_reduction_loss.item(),sim_loss.item(),loc_loss.item()))
        return rna_reduction_loss,st_reduction_loss,sim_loss,loc_loss,loss
    
class CellLoss(nn.Module):
    def __init__(self):
        super(CellLoss, self).__init__()

    def forward(self, rna_cell_out, rna_cell_label):
        rna_cell_loss = F.cross_entropy(rna_cell_out, rna_cell_label.long())
        return rna_cell_loss

                
                    