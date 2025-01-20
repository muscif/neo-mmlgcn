import torch
from torch_geometric.nn import LightGCN
from tabulate import tabulate
import GPUtil

from datetime import datetime
import random

from utils import train, test, CONFIG, MMDataset, print_config, EarlyStop
from mmlgcn import EF_MMLGCN, LF_MMLGCN, IF_MMLGCN


def main():
    gpu_id = GPUtil.getAvailable(order="memory", limit=10)[0]

    CONFIG.device = f"cuda:{gpu_id}"

    torch.cuda.manual_seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)

    fusion_types = {
        "early": EF_MMLGCN,
        "late": LF_MMLGCN,
        "inner": IF_MMLGCN,
    }

    Model = fusion_types[CONFIG.fusion_type]

    dataset = MMDataset()
    hdata = dataset.data

    num_users = hdata["user"].num_nodes
    num_items = hdata["item"].num_nodes
    data = hdata.to_homogeneous().to(CONFIG.device)

    # Use all message passing edges as training labels
    mask = data.edge_index[0] < data.edge_index[1]
    train_edge_label_index = data.edge_index[:, mask]
    
    size = train_edge_label_index.size(1)
    train_loader = torch.utils.data.DataLoader(
        range(size),
        shuffle=True,
        batch_size=CONFIG.batch_size,
        pin_memory=True,
        pin_memory_device=CONFIG.device,
    )

    if CONFIG.multimodal:
        model = Model(
            num_nodes=data.num_nodes,
            num_layers=CONFIG.n_layers,
            pretrained_modality_embeddings=dataset.embeddings,
        ).to(CONFIG.device)
    else:
        model = LightGCN(
            num_nodes=data.num_nodes,
            embedding_dim=CONFIG.embedding_dim,
            num_layers=CONFIG.n_layers,
        ).to(CONFIG.device)

    if torch.cuda.get_device_capability()[0] >= 7:
        model = torch.compile(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG.learning_rate, fused=True
    )

    out = []
    out.append(CONFIG)
    headers = ["Epoch", "Loss"]

    for k in CONFIG.top_k:
        headers.extend([f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"])

    out.append(headers)

    print_config()

    early_stop = EarlyStop(CONFIG.early_stop_window, CONFIG.early_stop_threshold)

    for epoch in range(CONFIG.epochs):
        loss = train(
            train_loader,
            train_edge_label_index,
            num_users,
            num_items,
            optimizer,
            model,
            data,
        )

        res = test(model, data, num_users, train_edge_label_index)

        metrics = [epoch + 1, round(loss.item(), 4)]

        for k in CONFIG.top_k:
            precision, recall, ndcg = res[k]
            metrics.extend([round(precision, 4), round(recall, 4), round(ndcg, 4)])

        out.append(metrics)

        print(tabulate([headers, metrics], tablefmt="plain"))
        print()

        if early_stop.is_stop([l[1] for l in out[2:]]):
            break

    if CONFIG.log:
        dt = datetime.now().replace(microsecond=0).isoformat()
        with open(f"logs/{dt}.log", "w", encoding="utf-8") as fout:
            conf = out[0]
            header = out[1]

            fout.write(str(conf) + "\n")
            fout.write('\t'.join(header) + "\n")

            for el in out[2:]:
                fout.write('\t'.join([str(e) for e in el]) + "\n")


if __name__ == "__main__":
    main()
