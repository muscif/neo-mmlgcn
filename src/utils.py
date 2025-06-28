import json
from pathlib import Path

import numpy as np
import toml
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree
from tqdm import tqdm
from tabulate import tabulate

PATH_CONFIG = Path("./config.toml")
PATH_LOG = Path("../logs")
PATH_DATA = Path("../data")


class Config:
    def __init__(self, config_path: Path):
        with open(config_path, "r") as file:
            config_data = toml.load(file)

        setattr(self, "device", None)

        for key, value in config_data.items():
            setattr(self, key, value)

    def __repr__(self):
        return json.dumps(self.__dict__)


CONFIG = Config(PATH_CONFIG)


def load_embeddings(path: Path):
    embeddings = {}

    for modality in CONFIG.datasets[CONFIG.dataset]:
        emb = torch.from_numpy(np.load(path / modality / "items.npy")).to(torch.float)

        embeddings[modality] = emb

    return embeddings

def create_edge_index(interactions, num_users):
    rows, cols = [], []
    for user, items in interactions:
        for item in items:
            rows.append(int(user))
            cols.append(int(item) + num_users)

    edge_index = torch.tensor([rows, cols])

    return edge_index

def transform(line):
    t = line.strip().split(" ")
    user, *items = t

    return user, items

def create_validation_split(path: Path):
    with (
        open(path / "train.txt", "r", encoding="utf-8") as fin_train,
        open(path / "test.txt", "r", encoding="utf-8") as fin_test
    ):
        train = [transform(line) for line in fin_train]
        test = [transform(line) for line in fin_test]

    users = set(int(user) for user, _ in train + test)

    num_users = len(users)

    edge_train = create_edge_index(train, num_users)

    size = edge_train.size(1)
    num_train = int(0.8 * size)
    shuffled_indices = torch.randperm(size)

    train_indices = shuffled_indices[:num_train]
    val_indices = shuffled_indices[num_train:]

    full_train_edge_index = edge_train.clone()
    train_edge_index = full_train_edge_index[:, train_indices]
    val_edge_label_index = full_train_edge_index[:, val_indices]

    return train_edge_index, val_edge_label_index, num_users

def load_dataset_new(path: Path):
    with (
        open(path / "train.txt", "r", encoding="utf-8") as fin_train,
        open(path / "test.txt", "r", encoding="utf-8") as fin_test,
        open(path / "validation.txt", "r", encoding="utf-8") as fin_val,
    ):
        train = [transform(line) for line in fin_train]
        val = [transform(line) for line in fin_val]
        test = [transform(line) for line in fin_test]

    users = set(int(user) for user, _ in train + val + test)
    items = set(int(item) for _, iteml in train + val + test for item in iteml)

    num_users = len(users)
    num_items = len(items)

    edge_train = create_edge_index(train, num_users)
    edge_val = create_edge_index(val, num_users)
    edge_test = create_edge_index(test, num_users)

    return {
        "num_users": num_users,
        "num_items": num_items,
        "edge_train": edge_train,
        "edge_val": edge_val,
        "edge_test": edge_test
    }

def train(
    train_loader, edge_train, num_users, num_items, optimizer, model
):
    model.train()
    total_loss = total_examples = 0

    for index in tqdm(train_loader):
        # Sample positive and negative labels.
        pos_edge_label_index = edge_train[:, index]
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

        optimizer.zero_grad(set_to_none=True)
        pos_rank, neg_rank = model(edge_train, edge_label_index).chunk(2)

        rec_loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )
        rec_loss.backward()

        optimizer.step()

        numel = pos_rank.numel()

        total_loss += rec_loss * numel
        total_examples += numel

    return total_loss / total_examples


@torch.no_grad()
def validate(val_loader, edge_val, num_users, num_items, model, edge_train):
    model.eval()
    total_loss = total_examples = 0

    for index in tqdm(val_loader):
        # Sample positive and negative labels.
        pos_edge_label_index = edge_val[:, index]
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

        pos_rank, neg_rank = model(edge_train, edge_label_index).chunk(2)

        rec_loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )

        numel = pos_rank.numel()

        total_loss += rec_loss * numel
        total_examples += numel

    return total_loss / total_examples


@torch.no_grad()
def test(
    model, num_users, edge_train, edge_test, edge_train_full
):
    emb = model.get_embedding(edge_train)
    user_emb, item_emb = emb[:num_users], emb[num_users:]
    res = {}

    for k in CONFIG.top_k:
        precision = recall = ndcg = total_examples = 0

        for start in range(0, num_users, CONFIG.batch_size):
            end = start + CONFIG.batch_size
            logits = user_emb[start:end] @ item_emb.t()

            # Exclude training edges
            mask = (edge_train_full[0] >= start) & (edge_train_full[0] < end)
            logits[
               edge_train_full[0, mask] - start,
               edge_train_full[1, mask] - num_users,
            ] = float("-inf")

            # Evaluate against test set
            ground_truth = torch.zeros_like(logits, dtype=torch.bool)
            mask = (edge_test[0] >= start) & (
                edge_test[0] < end
            )
            ground_truth[
                edge_test[0, mask] - start,
                edge_test[1, mask] - num_users,
            ] = True

            node_count = degree(
                edge_test[0, mask] - start, num_nodes=logits.size(0)
            )

            # Get top-k predictions
            topk_index = logits.topk(k, dim=-1).indices
            isin_mat = ground_truth.gather(1, topk_index)

            # Calculate precision and recall
            precision += float((isin_mat.sum(dim=-1) / k).sum())
            recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())

            # Calculate NDCG
            num_relevant = torch.minimum(node_count, torch.tensor(k))
            ideal_positions = torch.arange(k, device=CONFIG.device)
            dcg_weights = 1 / torch.log2(ideal_positions + 2)

            dcg = (isin_mat * dcg_weights).sum(dim=-1)

            idcg = torch.zeros_like(dcg)
            mask = (
                torch.arange(k, device=CONFIG.device)[None, :] < num_relevant[:, None]
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