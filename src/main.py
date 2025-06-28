import torch
from torch_geometric.nn import LightGCN
from tabulate import tabulate
import GPUtil

from datetime import datetime
import random
from zoneinfo import ZoneInfo

from utils import (
    train,
    test,
    validate,
    CONFIG,
    print_config,
    load_dataset_new,
    load_embeddings,
    PATH_LOG,
    PATH_DATA
)
from mmlgcn import EF_MMLGCN, LF_MMLGCN, IF_MMLGCN


def main(gpu_id):
    CONFIG.device = f"cuda:{gpu_id}"

    torch.cuda.manual_seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    random.seed(CONFIG.seed)

    data = load_dataset_new(PATH_DATA / CONFIG.dataset)
    
    edge_train = data["edge_train"].to(CONFIG.device)
    edge_val = data["edge_val"].to(CONFIG.device)
    edge_test = data["edge_test"].to(CONFIG.device)
    edge_train_full = torch.cat([edge_train, edge_val], dim=-1)

    # Create DataLoader for training and validation
    train_loader = torch.utils.data.DataLoader(
        range(edge_train.size(1)),
        shuffle=True,
        batch_size=CONFIG.batch_size,
        pin_memory=True,
        pin_memory_device=CONFIG.device,
    )

    val_loader = torch.utils.data.DataLoader(
        range(edge_val.size(1)),
        shuffle=True,
        batch_size=CONFIG.batch_size,
        pin_memory=True,
        pin_memory_device=CONFIG.device,
    )

    num_users = data["num_users"]
    num_items = data["num_items"]

    num_nodes = num_users + num_items

    if CONFIG.multimodal:
        fusion_types = {
            "early": EF_MMLGCN,
            "late": LF_MMLGCN,
            "inner": IF_MMLGCN,
        }

        embeddings = load_embeddings(PATH_DATA / CONFIG.dataset)

        Model = fusion_types[CONFIG.fusion_type]

        model = Model(
            num_nodes=num_nodes,
            num_layers=CONFIG.n_layers,
            pretrained_modality_embeddings=embeddings,
        ).to(CONFIG.device)
    else:
        model = LightGCN(
            num_nodes=num_nodes,
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
            edge_train,
            num_users,
            num_items,
            optimizer,
            model
        )

        val_loss = validate(
            val_loader, edge_val, num_users, num_items, model, edge_train
        )

        res = test(model, num_users, edge_train, edge_test, edge_train_full)

        metrics = [epoch + 1, round(train_loss.item(), 4), round(val_loss.item(), 4)]

        for k in CONFIG.top_k:
            precision, recall, ndcg = res[k]
            metrics.extend([round(precision, 4), round(recall, 4), round(ndcg, 4)])

        out.append(metrics)

        print(tabulate([headers, metrics], tablefmt="plain"))
        print()

    if CONFIG.log:
        dt = datetime.now(ZoneInfo("Europe/Rome")).replace(microsecond=0).isoformat()
        conf = out[0]
        header = out[1]

        with open(PATH_LOG / f"{dt}.log", "w", encoding="utf-8") as fout:
            if CONFIG.weighting == "alpha":
                fout.write(str(conf) + f"alpha: {model.weight}\n")
            else:
                fout.write(str(conf) + "\n")

            fout.write("\t".join(header) + "\n")

            for el in out[2:]:
                fout.write("\t".join([str(e) for e in el]) + "\n")

    print_config()


if __name__ == "__main__":
    gpu_id = GPUtil.getAvailable(order="memory", limit=10, maxLoad=1, maxMemory=1)[0]
    main(gpu_id)
