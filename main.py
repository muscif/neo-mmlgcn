import torch
from torch_geometric.nn import LightGCN
from tabulate import tabulate

from datetime import datetime
import random

from utils import train, test, CONFIG, MMDataset, print_config, is_early_stop
from mmlgcn import EF_MMLGCN, LF_MMLGCN, IF_MMLGCN

torch.manual_seed(CONFIG.seed)
random.seed(CONFIG.seed)

models = {
    "ef-mmlgcn": EF_MMLGCN,
    "lf-mmlgcn": LF_MMLGCN,
    "if-mmlgcn": IF_MMLGCN
}

Model = models[CONFIG.model]

dataset = MMDataset()
hdata = dataset.data

num_users = hdata["user"].num_nodes
num_items = hdata["item"].num_nodes
data = hdata.to_homogeneous().to(CONFIG.device)

# Use all message passing edges as training labels
mask = data.edge_index[0] < data.edge_index[1]
train_edge_label_index = data.edge_index[:, mask]
train_loader = torch.utils.data.DataLoader(
    range(train_edge_label_index.size(1)),
    shuffle=True,
    batch_size=CONFIG.batch_size,
)

if CONFIG.multimodal:
    model = Model(
        num_nodes=data.num_nodes,
        num_layers=CONFIG.n_layers,
        pretrained_modality_embeddings=dataset.embeddings
    ).to(CONFIG.device)
else:
    model = LightGCN(
        num_nodes=data.num_nodes,
        embedding_dim=CONFIG.embedding_dim,
        num_layers=CONFIG.n_layers,
    ).to(CONFIG.device)

optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)

out = []
out.append(CONFIG)
headers = ["Epoch", "Loss"]

for k in CONFIG.top_k:
    headers.extend([f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"])

out.append(headers)

print_config()

for epoch in range(CONFIG.epochs):
    loss = train(
        train_loader,
        train_edge_label_index,
        num_users,
        num_items,
        optimizer,
        model,
        data
    )

    res = test(
        model,
        data,
        num_users,
        train_edge_label_index
    )

    metrics = [epoch + 1, round(loss, 4)]

    for k in CONFIG.top_k:
        precision, recall, ndcg = res[k]
        metrics.extend([round(precision, 4), round(recall, 4), round(ndcg, 4)])

    out.append(metrics)

    prev_losses = [l[1] for l in out[2:][-CONFIG.early_stop_n:]]
    if is_early_stop(prev_losses):
        break

    print(tabulate([headers, metrics], tablefmt="plain"))
    print()

exit()
dt = datetime.now().replace(microsecond=0).isoformat()
with open(f"logs/{dt}.log", "w", encoding="utf-8") as fout:
    conf = out[0]
    header = out[1]

    fout.write(f"{conf}\n")
    fout.write(f"{'\t'.join(header)}\n")

    for el in out[2:]:
        fout.write(f"{'\t'.join([str(e) for e in el])}\n")

print_config()