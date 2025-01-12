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
    num_modalities = stacked_embeddings.shape[0]

    concatenated_embeddings = torch.cat(
        [stacked_embeddings[i] for i in range(num_modalities)], dim=-1
    ).to(CONFIG.device)

    projector = nn.Linear(
        num_modalities * CONFIG.embedding_dim, CONFIG.embedding_dim
    ).to(CONFIG.device)

    aggregated_embeddings = projector(concatenated_embeddings)

    return aggregated_embeddings


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
    "mean": fuse_mean,
    "sum": fuse_sum,
    "max": fuse_max,
    "min": fuse_min,
    "prod": fuse_prod,
}

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
        pt_mm_emb = [
            F.normalize(emb) for emb in pretrained_modality_embeddings.values()
        ]

        self.mm_embeddings = emb_fn[CONFIG.single_branch](pt_mm_emb)

        self.num_items = len(self.mm_embeddings[0])
        self.num_users = num_nodes - self.num_items
        self.fuse = fusion_fn[CONFIG.fusion_modalities]

    def get_embedding(self, edge_index, edge_weight=None):
        lgcn_emb = super().get_embedding(edge_index, edge_weight)

        user_emb = lgcn_emb[: self.num_users]
        item_emb = lgcn_emb[self.num_users :]

        return user_emb, item_emb


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

        stacked_embeddings = torch.stack(self.mm_embeddings).to(CONFIG.device)

        fused_mm_embeddings = self.fuse(stacked_embeddings)

        self.fused_mm_embeddings = nn.Embedding.from_pretrained(
            fused_mm_embeddings, freeze=CONFIG.freeze
        )

    def get_embedding(self, edge_index, edge_weight=None):
        user_emb, item_emb = super().get_embedding(edge_index, edge_weight)

        final_item_emb = fuse_mean(
            torch.stack([item_emb, self.fused_mm_embeddings.weight])
        )

        final_emb = torch.cat([user_emb, final_item_emb], dim=0)

        return final_emb


class LF_MMLGCN(Base_MMLGCN):
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

        self.mm_embeddings = [
            nn.Embedding.from_pretrained(emb, freeze=CONFIG.freeze).to(CONFIG.device)
            for emb in self.mm_embeddings
        ]

    def get_embedding(self, edge_index, edge_weight=None):
        user_emb, item_emb = super().get_embedding(edge_index, edge_weight)

        fused_emb = torch.stack([item_emb, *[emb.weight for emb in self.mm_embeddings]])

        final_item_emb = self.fuse(fused_emb)

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

        stacked_embeddings = torch.stack(self.mm_embeddings)
        fused_mm_embeddings = self.fuse(stacked_embeddings)

        emb = torch.cat(
            [
                self.embedding.weight[: self.num_users].to(CONFIG.device),
                fused_mm_embeddings.to(CONFIG.device),
            ]
        )

        self.embedding = nn.Embedding.from_pretrained(emb, freeze=False)

    def get_embedding(self, edge_index, edge_weight=None):
        return super(Base_MMLGCN, self).get_embedding(edge_index, edge_weight)


class LMF_MMLGCN(Base_MMLGCN):
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

        stacked_embeddings = torch.stack(self.mm_embeddings)

        self.fused_mm_embeddings = [
            nn.Embedding.from_pretrained(
                f(stacked_embeddings).to(CONFIG.device), freeze=CONFIG.freeze
            )
            for f in fusion_fn.values()
        ]

    def get_embedding(self, edge_index, edge_weight=None):
        user_emb, item_emb = super().get_embedding(edge_index, edge_weight)

        stacked_item_emb = torch.stack(
            [item_emb, *[emb.weight for emb in self.fused_mm_embeddings]]
        )

        final_item_emb = self.fuse(stacked_item_emb)

        final_emb = torch.cat([user_emb, final_item_emb], dim=0)

        return final_emb


class EMF_MMLGCN(Base_MMLGCN):
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

        stacked_embeddings = torch.stack(self.mm_embeddings)

        fused_mm_embeddings = torch.stack(
            [f(stacked_embeddings) for f in fusion_fn.values()]
        )

        projector = nn.Sequential(nn.LazyLinear(CONFIG.embedding_dim // len(fusion_fn)))
        projected = torch.cat([projector(emb) for emb in fused_mm_embeddings], dim=-1)

        final_projector = nn.LazyLinear(CONFIG.embedding_dim)
        projected = final_projector(projected)

        self.fused_mm_embeddings = nn.Embedding.from_pretrained(
            projected, freeze=CONFIG.freeze
        )

    def get_embedding(self, edge_index, edge_weight=None):
        user_emb, item_emb = super().get_embedding(edge_index, edge_weight)

        stacked_embeddings = torch.stack([item_emb, self.fused_mm_embeddings.weight])

        final_item_emb = self.fuse(stacked_embeddings)

        final_emb = torch.cat([user_emb, final_item_emb], dim=0)

        return final_emb
