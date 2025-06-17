import torch
import torch.nn as nn
from torch_geometric.nn import LightGCN
from info_nce import InfoNCE
import torch.nn.functional as F
from functools import partial

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
        nn.LazyLinear(CONFIG.embedding_dim),
    )

    for modality_emb in pretrained_modality_embeddings:
        ll = nn.Sequential(
            nn.LazyLinear(CONFIG.embedding_dim),
        )
        emb = ll(modality_emb)

        mm_embeddings.append(emb)

    mm_embeddings = [sb_layer(emb) for emb in mm_embeddings]

    return mm_embeddings


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


def fuse_concat(stacked_embeddings, layer):
    concatenated = torch.cat([emb for emb in stacked_embeddings], dim=1).to(
        CONFIG.device
    )
    reduced = layer(concatenated)

    return reduced


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

        emb_fn = {False: get_mm_embeddings, True: get_mm_embeddings_single_branch}

        self.mm_embeddings = [
            nn.Embedding.from_pretrained(emb, freeze=CONFIG.freeze).to(CONFIG.device)
            for emb in emb_fn[CONFIG.single_branch](embs)
        ]

        self.num_items = len(self.mm_embeddings[0].weight)
        self.num_users = num_nodes - self.num_items

        if CONFIG.fusion_modalities == "concat":
            layer = nn.LazyLinear(CONFIG.embedding_dim, device=CONFIG.device)
            self.fuse = partial(fuse_concat, layer=layer)
        else:
            self.fuse = fusion_fn[CONFIG.fusion_modalities]

        if CONFIG.ensemble_fusion:
            self.stacked_embeddings = fuse_ensemble(
                torch.stack([mm_emb.weight for mm_emb in self.mm_embeddings])
            )

        self.mm_weight = (
            nn.Parameter(torch.tensor(0.0, device=CONFIG.device)) if CONFIG.alpha else 1
        )

        self.encoder = nn.Sequential(
            nn.LazyLinear(CONFIG.embedding_dim // 2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.LazyLinear(CONFIG.embedding_dim),
            nn.ReLU(),
        )

        self.info_nce_loss = InfoNCE()

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
                mod_weight = mod.weight
                encoded = self.encoder(mod_weight)
                decoded = self.decoder(encoded)

                mse_loss += F.mse_loss(mod_weight, decoded)

            loss += mse_loss * 300

        if CONFIG.info_nce:
            emb0, emb1 = self.mm_embeddings

            info_nce_loss = self.info_nce_loss(emb0.weight, emb1.weight)

            loss += info_nce_loss / 10

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

        self.mm_emb_list = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(mm_emb.weight, freeze=CONFIG.freeze)
                for mm_emb in self.mm_embeddings
            ]
        )
        self.n_modalities = len(self.mm_emb_list)

        stacked_embeddings = torch.stack([emb.weight for emb in self.mm_emb_list])

        self.fused_mm_embeddings = nn.Embedding.from_pretrained(
            self.fuse(stacked_embeddings), freeze=CONFIG.freeze
        )

    def get_embedding(self, edge_index, edge_weight=None):
        user_emb, item_emb = super().get_embedding(edge_index, edge_weight)
        final_item_emb = self.fuse(
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

        self.mm_emb_list = nn.ModuleList(
            [
                nn.Embedding.from_pretrained(mm_emb.weight, freeze=CONFIG.freeze)
                for mm_emb in self.mm_embeddings
            ]
        )

        self.n_modalities = len(self.mm_emb_list)

    def get_embedding(self, edge_index, edge_weight=None):
        user_emb, item_emb = super().get_embedding(edge_index, edge_weight)

        match CONFIG.weighting:
            case "alpha":
                alpha = F.normalize(self.mm_weight)
                stacked_item_emb = torch.stack(
                    [
                        item_emb * alpha,
                        *[emb.weight * (1 - alpha) for emb in self.mm_emb_list],
                    ]
                )
            case "normalized":
                stacked_item_emb = torch.stack(
                    [
                        F.normalize(item_emb) * self.n_modalities,
                        *[emb.weight for emb in self.mm_emb_list],
                    ]
                )
            case "equal":
                stacked_mm_emb = torch.stack([emb.weight for emb in self.mm_emb_list])
                stacked_mm_emb = self.fuse(stacked_mm_emb)
                stacked_item_emb = torch.stack([item_emb, stacked_mm_emb])
            case False:
                stacked_item_emb = torch.stack(
                    [item_emb, *[emb.weight for emb in self.mm_emb_list]]
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
