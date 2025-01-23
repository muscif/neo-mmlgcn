import json
import os.path as osp

import numpy as np
import toml
import torch
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

def load_embeddings(path):
    embeddings = {}

    for modality in CONFIG.datasets[CONFIG.dataset]:
        emb = torch.from_numpy(np.load(f"{path}/{modality}/items.npy")).to(
            torch.float
        )

        embeddings[modality] = emb

    return embeddings

def load_dataset(path):
    def transform(line):
        t = line.strip().split(" ")
        user, *items = t

        return user, items
    
    def create_edge_index(interactions):
        rows, cols = [], []
        for user, items in interactions:
            for item in items:
                rows.append(int(user))
                cols.append(int(item))

        edge_index = torch.tensor([rows, cols]) 

        return edge_index

    train_file = osp.join(path, "train.txt")
    test_file = osp.join(path, "test.txt")

    
    with (
        open(train_file, "r", encoding="utf-8") as fin_train,
        open(test_file, "r", encoding="utf-8") as fin_test,
    ):
        train = [transform(line) for line in fin_train]
        test = [transform(line) for line in fin_test]

    # PyG format: user IDs should range from (0, num_users) and item IDs should range from (num_users, num_users + num_items)
    # https://github.com/pyg-team/pytorch_geometric/discussions/8788#discussioncomment-8197084
    # This doesn't work, raises CUDA index error

    users = set(int(user) for user, _ in train + test)
    items = set(int(item) for _, iteml in train + test for item in iteml)

    users_remap = {user: i for i, user in enumerate(users)}
    items_remap = {item: i for i, item in enumerate(items)}

    train_remap = [
        (users_remap[int(user)], [items_remap[int(item)] for item in iteml]) 
        for user, iteml in train
    ]

    test_remap = [
        (users_remap[int(user)], [items_remap[int(item)] for item in iteml]) 
        for user, iteml in test
    ]
    
    data = HeteroData()

    data["user"].num_nodes = len(users)
    data["item"].num_nodes = len(items)

    data.num_nodes = data["user"].num_nodes + data["item"].num_nodes

    edge_index = create_edge_index(train_remap)
    edge_label_index = create_edge_index(test_remap)

    data["user", "rates", "item"].edge_index = edge_index
    data["user", "rates", "item"].edge_label_index = edge_label_index

    return data
    

def train(
    train_loader, train_edge_label_index, num_users, num_items, optimizer, model, data
):
    model.train()
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

        optimizer.zero_grad(set_to_none=True)
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
def validate(
    train_loader, train_edge_label_index, num_users, num_items, model, data
):
    model.eval()
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

        pos_rank, neg_rank = model(data.edge_index, edge_label_index).chunk(2)

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
