import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN
import torch.nn.functional as F

from utils import CONFIG


def get_mm_embeddings(pretrained_modality_embeddings):
    mm_embeddings = []

    for emb in pretrained_modality_embeddings:
        ll = nn.Sequential(
            nn.LazyLinear(CONFIG.embedding_dim),
        )
        emb = ll(emb)

        mm_embeddings.append(emb)

    return mm_embeddings


def get_mm_embeddings_single_branch(pretrained_modality_embeddings):
    mm_embeddings = []

    sb_layer = nn.Sequential(
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


def fuse_concat(stacked_embeddings):
    conc = torch.cat([emb for emb in stacked_embeddings], dim=-1)
    projector = nn.LazyLinear(CONFIG.embedding_dim)
    fused_embeddings = projector(conc)

    return fused_embeddings


def fuse_mean(stacked_embeddings):
    return torch.mean(stacked_embeddings, dim=0)


def fuse_sum(stacked_embeddings):
    return torch.sum(stacked_embeddings, dim=0)


def fuse_max(stacked_embeddings):
    return torch.max(stacked_embeddings, dim=0).values


def fuse_min(stacked_embeddings):
    return torch.min(stacked_embeddings, dim=0).values


def fuse_prod(stacked_embeddings):
    return torch.prod(stacked_embeddings, dim=0)


fusion_fn = {
    "concat": fuse_concat,
    "mean": fuse_mean,
    "sum": fuse_sum,
    "max": fuse_max,
    "min": fuse_min,
    "prod": fuse_prod,
}


def fuse_ensemble(stacked_embeddings):
    return torch.stack([f(stacked_embeddings) for f in fusion_fn.values()])


emb_fn = {False: get_mm_embeddings, True: get_mm_embeddings_single_branch}


class Base_MMLGCN(LightGCN):
    def __init__(
        self,
        num_nodes,
        num_layers,
        pretrained_modality_embeddings,
        alpha=None,
        **kwargs,
    ):
        super().__init__(num_nodes, CONFIG.embedding_dim, num_layers, alpha, **kwargs)
        embs = [F.normalize(emb) for emb in pretrained_modality_embeddings.values()]

        self.mm_embeddings = [
            nn.Embedding.from_pretrained(emb, freeze=CONFIG.freeze)
            for emb in emb_fn[CONFIG.single_branch](embs)
        ]

        self.num_items = len(self.mm_embeddings[0].weight)
        self.num_users = num_nodes - self.num_items
        self.fuse = fusion_fn[CONFIG.fusion_modalities]

        self.stacked_embeddings = torch.stack(
            [emb.weight for emb in self.mm_embeddings]
        )

        if CONFIG.ensemble_fusion:
            self.stacked_embeddings = fuse_ensemble(self.stacked_embeddings)

        self.encoder = nn.Sequential(
            nn.LazyLinear(CONFIG.embedding_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.LazyLinear(CONFIG.embedding_dim),
            nn.ReLU(),
        )

    def get_embedding(self, edge_index, edge_weight=None):
        lgcn_emb = super().get_embedding(edge_index, edge_weight)

        user_emb = lgcn_emb[: self.num_users]
        item_emb = lgcn_emb[self.num_users :]

        return user_emb, item_emb

    def recommendation_loss(
        self, pos_edge_rank, neg_edge_rank, node_id=None, lambda_reg=0.0001, **kwargs
    ):
        loss = super().recommendation_loss(
            pos_edge_rank, neg_edge_rank, node_id, lambda_reg, **kwargs
        )

        if CONFIG.autoencoder:
            mse_loss = 0

            for mod in self.mm_embeddings:
                encoded = self.encoder(mod)
                decoded = self.decoder(encoded)

                mse_loss += F.mse_loss(mod, decoded)

            loss += mse_loss * 200

        return loss


class EF_MMLGCN(Base_MMLGCN):
    def __init__(
        self,
        num_nodes,
        num_layers,
        pretrained_modality_embeddings,
        alpha=None,
        **kwargs,
    ):
        super().__init__(
            num_nodes, num_layers, pretrained_modality_embeddings, alpha, **kwargs
        )

        self.fused_mm_embeddings = nn.Embedding.from_pretrained(
            self.fuse(self.stacked_embeddings), freeze=CONFIG.freeze
        )

    def get_embedding(self, edge_index, edge_weight=None):
        user_emb, item_emb = super().get_embedding(edge_index, edge_weight)

        final_item_emb = fuse_mean(
            torch.stack([item_emb, self.fused_mm_embeddings.weight])
        )

        final_emb = torch.cat([user_emb, final_item_emb], dim=0)

        return final_emb


class LF_MMLGCN(Base_MMLGCN):
    def get_embedding(self, edge_index, edge_weight=None):
        user_emb, item_emb = super().get_embedding(edge_index, edge_weight)

        stacked_item_emb = torch.stack(
            [item_emb, *[emb.weight for emb in self.mm_embeddings]]
        )

        final_item_emb = self.fuse(stacked_item_emb)

        return torch.cat([user_emb, final_item_emb], dim=0)


class IF_MMLGCN(Base_MMLGCN):
    def __init__(
        self,
        num_nodes,
        num_layers,
        pretrained_modality_embeddings,
        alpha=None,
        **kwargs,
    ):
        super().__init__(
            num_nodes, num_layers, pretrained_modality_embeddings, alpha, **kwargs
        )

        fused_mm_embeddings = self.fuse(self.stacked_embeddings)

        emb = torch.cat(
            [
                self.embedding.weight[: self.num_users],
                fused_mm_embeddings,
            ]
        )

        self.embedding = nn.Embedding.from_pretrained(emb, freeze=False)

    def get_embedding(self, edge_index, edge_weight=None):
        return super(Base_MMLGCN, self).get_embedding(edge_index, edge_weight)
