import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN
from torch.nn import Embedding

from utils import CONFIG, attention


def fuse_concat(stacked_embeddings):
    num_modalities, _, d = stacked_embeddings.shape

    concatenated_embeddings = torch.cat([stacked_embeddings[i] for i in range(num_modalities)], dim=-1).to(CONFIG.device) # [num_items, num_modalities * d]

    projector = nn.Linear(num_modalities * d, CONFIG.embedding_dim).to(CONFIG.device)

    aggregated_embeddings = projector(concatenated_embeddings) # [num_items, embedding_dim]

    return aggregated_embeddings

def fuse_mean(stacked_embeddings):
    return torch.mean(stacked_embeddings, dim=0)

def fuse_sum(stacked_embeddings):
    return torch.sum(stacked_embeddings, dim=0)

def fuse_max(stacked_embeddings):
    return torch.max(stacked_embeddings, dim=0).values

def fuse_min(stacked_embeddings):
    return torch.min(stacked_embeddings, dim=0).values

def fuse_attention(stacked_embeddings):
    return fuse_mean(attention(stacked_embeddings, stacked_embeddings, stacked_embeddings))

def fuse(stacked_embeddings):
    function = None

    match CONFIG.fusion:
        case "concat":
            function = fuse_concat
        case "mean":
            function = fuse_mean
        case "sum":
            function = fuse_sum
        case "max":
            function = fuse_max
        case "min":
            function = fuse_min
        case "attention":
            function = fuse_attention

    return function(stacked_embeddings)

def get_mm_embeddings(pretrained_modality_embeddings):
    mm_embeddings = []

    for modality_emb in pretrained_modality_embeddings:
        ll = nn.Sequential(
            nn.Linear(modality_emb.shape[1], CONFIG.embedding_dim),
        )
        emb = ll(modality_emb)
        
        mm_embeddings.append(emb)

    return mm_embeddings

# Single branch
def get_mm_embeddings_sb(pretrained_modality_embeddings: list):
    mm_embeddings = []

    sb_layer = nn.Sequential(
        nn.ReLU(),
        nn.Linear(CONFIG.embedding_dim * 2, CONFIG.embedding_dim),
    )

    for modality_emb in pretrained_modality_embeddings:
        ll = nn.Sequential(
            nn.Linear(modality_emb.shape[1], CONFIG.embedding_dim * 2),
        )
        emb = ll(modality_emb)
        mm_embeddings.append(emb)

    mm_embeddings = [sb_layer(emb) for emb in mm_embeddings]

    return mm_embeddings

class EF_MMLGCN(LightGCN):
    def __init__(self, num_nodes, num_layers, pretrained_modality_embeddings, alpha = None, **kwargs):
        super().__init__(num_nodes, CONFIG.embedding_dim, num_layers, alpha, **kwargs)
        pt_mm_emb = list(pretrained_modality_embeddings.values())

        if CONFIG.single_branch:
            mm_embeddings = get_mm_embeddings_sb(pt_mm_emb)
        else:
            mm_embeddings = get_mm_embeddings(pt_mm_emb)

        self.num_items = len(mm_embeddings[0])
        self.num_users = num_nodes - self.num_items

        stacked_embeddings = torch.stack(mm_embeddings)
        fused_mm_embeddings = fuse(stacked_embeddings)

        self.register_buffer("fused_mm_embeddings", fused_mm_embeddings.detach())
        self.reset_parameters()

    def get_embedding(self, edge_index, edge_weight=None):
        lgcn_emb = super().get_embedding(edge_index, edge_weight)
        fused_mm_embeddings = self.get_buffer("fused_mm_embeddings")

        user_emb = lgcn_emb[:self.num_users]
        item_emb = lgcn_emb[self.num_users:]

        final_item_emb = fuse_mean(torch.stack([item_emb, fused_mm_embeddings]))
        
        final_emb = torch.cat([user_emb, final_item_emb], dim=0)

        return final_emb

class LF_MMLGCN(LightGCN):
    def __init__(self, num_nodes, num_layers, pretrained_modality_embeddings, alpha = None, **kwargs):
        super().__init__(num_nodes, CONFIG.embedding_dim, num_layers, alpha, **kwargs)
        mm_embeddings = get_mm_embeddings_sb(list(pretrained_modality_embeddings.values()))

        self.num_items = len(mm_embeddings[0])
        self.num_users = num_nodes - self.num_items

        stacked_embeddings = torch.stack(mm_embeddings)
        self.register_buffer("stacked_embeddings", stacked_embeddings.detach())
        self.reset_parameters()

    def get_embedding(self, edge_index, edge_weight=None):
        lgcn_emb = super().get_embedding(edge_index, edge_weight)
        stacked_embeddings = self.get_buffer("stacked_embeddings")

        user_emb = lgcn_emb[:self.num_users]
        item_emb = lgcn_emb[self.num_users:]

        mm_emb = torch.cat([item_emb.unsqueeze(0), stacked_embeddings])

        final_item_emb = fuse(mm_emb)

        return torch.cat([user_emb, final_item_emb], dim=0)

# Pretrained initialization
class IF_MMLGCN(LightGCN):
    def __init__(self, num_nodes, num_layers, pretrained_modality_embeddings, alpha = None, **kwargs):
        mm_embeddings = get_mm_embeddings(pretrained_modality_embeddings)
        
        self.num_items = len(mm_embeddings[0])
        self.num_users = num_nodes - self.num_items

        super().__init__(num_nodes, CONFIG.embedding_dim, num_layers, alpha, **kwargs)

        stacked_embeddings = torch.stack(mm_embeddings)
        fused_mm_embeddings = fuse(stacked_embeddings)

        emb = torch.cat([self.embedding.weight[:self.num_users].to(CONFIG.device), fused_mm_embeddings.to(CONFIG.device)])
        self.embedding = Embedding.from_pretrained(emb, freeze=False)
        #self.embedding.weight[self.num_users].detach() # Should freeze item embeddings

    # Debugging function to check if item embeddings are frozen
    def get_embedding(self, edge_index, edge_weight = None):
        lgcn_emb = super().get_embedding(edge_index, edge_weight)

        user_emb = lgcn_emb[:self.num_users]
        item_emb = lgcn_emb[self.num_users:]

        return lgcn_emb