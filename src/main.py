import torch
from torch_geometric.nn import LightGCN
from tabulate import tabulate
import GPUtil
from hashlib import md5

from datetime import datetime
import random

from utils import (
    train,
    test,
    validate,
    CONFIG,
    print_config,
    load_dataset,
    load_embeddings,
    PATH_LOG,
    PATH_DATA
)
from mmlgcn import EF_MMLGCN, LF_MMLGCN, IF_MMLGCN


def main():
    gpu_id = GPUtil.getAvailable(order="memory", limit=10, maxLoad=1, maxMemory=1)[0]

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

    
    hdata = load_dataset(PATH_DATA / CONFIG.dataset)
    embeddings = load_embeddings(PATH_DATA / CONFIG.dataset)

    num_users = hdata["user"].num_nodes
    num_items = hdata["item"].num_nodes
    data = hdata.to_homogeneous().to(CONFIG.device)

    train_edge_index = data.edge_index
    test_edge_label_index = data.edge_label_index

    size = train_edge_index.size(1)
    num_train = int(0.8 * size)
    shuffled_indices = torch.randperm(size)
    train_indices = shuffled_indices[:num_train]
    val_indices = shuffled_indices[num_train:]

    full_train_edge_index = train_edge_index.clone()
    train_edge_index = full_train_edge_index[:, train_indices]
    val_edge_label_index = full_train_edge_index[:, val_indices]

    # Create DataLoader for training and validation
    train_loader = torch.utils.data.DataLoader(
        range(train_edge_index.size(1)),
        shuffle=True,
        batch_size=CONFIG.batch_size,
        pin_memory=True,
        pin_memory_device=CONFIG.device,
    )

    val_loader = torch.utils.data.DataLoader(
        range(val_edge_label_index.size(1)),
        shuffle=True,
        batch_size=CONFIG.batch_size,
        pin_memory=True,
        pin_memory_device=CONFIG.device,
    )

    if CONFIG.multimodal:
        model = Model(
            num_nodes=data.num_nodes,
            num_layers=CONFIG.n_layers,
            pretrained_modality_embeddings=embeddings,
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
    headers = ["Epoch", "T-Loss", "V-Loss"]

    for k in CONFIG.top_k:
        headers.extend([f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}"])

    out.append(headers)

    print_config()

    for epoch in range(CONFIG.epochs):
        train_loss = train(
            train_loader,
            train_edge_index,
            num_users,
            num_items,
            optimizer,
            model,
            data,
        )

        val_loss = validate(
            val_loader, val_edge_label_index, num_users, num_items, model, data
        )

        res = test(model, num_users, train_edge_index, test_edge_label_index, full_train_edge_index)

        metrics = [epoch + 1, round(train_loss.item(), 4), round(val_loss.item(), 4)]

        for k in CONFIG.top_k:
            precision, recall, ndcg = res[k]
            metrics.extend([round(precision, 4), round(recall, 4), round(ndcg, 4)])

        out.append(metrics)

        print(tabulate([headers, metrics], tablefmt="plain"))
        print()

    if CONFIG.log:
        #dt = datetime.now().replace(microsecond=0).isoformat()
        conf = out[0]
        header = out[1]

        conf_hash = md5(bytes(str(conf))).hexdigest()

        with open(PATH_LOG / f"{conf_hash}.log", "w", encoding="utf-8") as fout:
            if CONFIG.alpha:
                fout.write(str(conf) + f"alpha: {model.weight}" + "\n")
            else:
                fout.write(str(conf) + "\n")

            fout.write("\t".join(header) + "\n")

            for el in out[2:]:
                fout.write("\t".join([str(e) for e in el]) + "\n")

    print_config()


if __name__ == "__main__":
    main()
