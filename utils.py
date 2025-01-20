import json
import os.path as osp

import numpy as np
from numpy.polynomial import Polynomial
import toml
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
from tqdm import tqdm
from tabulate import tabulate


class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as file:
            config_data = toml.load(file)

        setattr(self, "device", None)

        for key, value in config_data.items():
            setattr(self, key, value)

    def __repr__(self):
        return json.dumps(self.__dict__)


CONFIG = Config("config.toml")


class MMDataset:
    def __init__(self):
        self.path = f"./data/{CONFIG.dataset}"

        self.load_embeddings()
        self.load_dataset()

    def load_embeddings(self):
        self.embeddings = {}

        for modality in CONFIG.datasets[CONFIG.dataset]:
            emb = torch.from_numpy(np.load(f"{self.path}/{modality}/items.npy")).to(
                torch.float
            )

            self.embeddings[modality] = emb

    def load_dataset(self):
        def transform(line):
            t = line.strip().split(" ")
            user, *items = t

            return user, items

        train_file = osp.join(self.path, "train.txt")
        test_file = osp.join(self.path, "test.txt")

        self.data = HeteroData()
        with (
            open(train_file, "r", encoding="utf-8") as fin_train,
            open(test_file, "r", encoding="utf-8") as fin_test,
        ):
            train = [transform(line) for line in fin_train]
            test = [transform(line) for line in fin_test]

        items = set()
        users = set()
        for user, its in train + test:
            users.add(user)

            for it in its:
                items.add(it)

        self.data["user"].num_nodes = len(users)
        self.data["item"].num_nodes = len(items)

        # Process edge information for training and testing:
        attr_names = ["edge_index", "edge_label_index"]
        for portion, attr_name in zip([train, test], attr_names):
            rows, cols = [], []

            for user, items in portion:
                for dst in items:
                    rows.append(int(user))
                    cols.append(int(dst))
            index = torch.tensor([rows, cols])

            self.data["user", "rates", "item"][attr_name] = index

            if CONFIG.bidirectional:
                if attr_name == "edge_index":
                    self.data["item", "rated_by", "user"][attr_name] = index.flip([0])


class Attention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super().__init__()
        self.kq = nn.Linear(embedding_dim, attention_dim).to(CONFIG.device)
        self.v = nn.Linear(embedding_dim, embedding_dim).to(CONFIG.device)

    def forward(self, q, k, v):
        q = self.kq(q)
        k = self.kq(k)
        v = self.v(v)

        scores = q @ torch.transpose(k, 1, 2)
        scores = scores / (k.shape[2] ** 0.5)
        scores = nn.functional.softmax(scores, dim=2)
        scores = scores @ v

        return scores.squeeze(1)


class SelfAttention(Attention):
    def forward(self, q):
        return super().forward(q, q, q)


class CrossAttention(Attention):
    def forward(self, t1, t2):
        return super().forward(t1, t2, t2)


class EarlyStop:
    def __init__(self, window=20, threshold=0.01):
        self.window = window
        self.threshold = threshold
        self.x = np.arange(window)

    def is_stop(self, l):
        if len(l) > self.window:
            slope, _ = Polynomial.fit(self.x, l[-self.window:], 1)
            return abs(slope) < self.threshold

@torch.compile
def train(
    train_loader, train_edge_label_index, num_users, num_items, optimizer, model, data
):
    total_loss = total_examples = 0

    for index in tqdm(train_loader):
        # Sample positive and negative labels.
        pos_edge_label_index = train_edge_label_index[:, index]
        neg_edge_label_index = torch.stack(
            [
                pos_edge_label_index[0],
                torch.randint(
                    num_users,
                    num_users + num_items,
                    (index.numel(),),
                    device=CONFIG.device,
                ),
            ],
            dim=0,
        )
        edge_label_index = torch.cat(
            [
                pos_edge_label_index,
                neg_edge_label_index,
            ],
            dim=1,
        )

        optimizer.zero_grad()
        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)

        rec_loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        rec_loss.backward(retain_graph=True)

        optimizer.step()

        numel = pos_rank.numel()

        total_loss += rec_loss * numel
        total_examples += numel

    return total_loss / total_examples


@torch.no_grad()
def test(model, data, num_users, train_edge_label_index):
    emb = model.get_embedding(data.edge_index)
    user_emb, item_emb = emb[:num_users], emb[num_users:]
    res = {}

    for k in CONFIG.top_k:
        precision = recall = ndcg = total_examples = 0

        for start in range(0, num_users, CONFIG.batch_size):
            end = start + CONFIG.batch_size
            logits = user_emb[start:end] @ item_emb.t()

            # Exclude training edges
            mask = (train_edge_label_index[0] >= start) & (
                train_edge_label_index[0] < end
            )
            logits[
                train_edge_label_index[0, mask] - start,
                train_edge_label_index[1, mask] - num_users,
            ] = float("-inf")

            # Computing ground truth
            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            mask = (data.edge_label_index[0] >= start) & (
                data.edge_label_index[0] < end
            )
            ground_truth[
                data.edge_label_index[0, mask] - start,
                data.edge_label_index[1, mask] - num_users,
            ] = True
            node_count = degree(
                data.edge_label_index[0, mask] - start, num_nodes=logits.size(0)
            )

            # Get top-k predictions
            topk_index = logits.topk(k, dim=-1).indices
            isin_mat = ground_truth.gather(1, topk_index)

            # Calculate precision and recall
            precision += float((isin_mat.sum(dim=-1) / k).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())

            # Calculate NDCG
            num_relevant = torch.minimum(node_count, torch.tensor(k))
            ideal_positions = torch.arange(k, device=logits.device)
            dcg_weights = 1 / torch.log2(ideal_positions + 2)

            dcg = (isin_mat * dcg_weights).sum(dim=-1)

            idcg = torch.zeros_like(dcg)
            mask = (
                torch.arange(k, device=logits.device)[None, :] < num_relevant[:, None]
            )
            idcg = (mask * dcg_weights[None, :]).sum(dim=1)

            # Compute NDCG and check whether IDCG is 0
            valid_mask = idcg > 0
            ndcg += float((dcg[valid_mask] / idcg[valid_mask]).sum())

            total_examples += int((node_count > 0).sum())

        precision = precision / total_examples
        recall = recall / total_examples
        ndcg = ndcg / total_examples

        res[k] = precision, recall, ndcg

    return res


def print_config():
    d = dict(CONFIG.__dict__)

    if "datasets" in d:
        d.pop("datasets")

    print("CONFIGURATION")
    print(tabulate(d.items()))
