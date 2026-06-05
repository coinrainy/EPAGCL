import json
import os
import random

import torch
import numpy as np

from tqdm import tqdm
from time import time
from torch_geometric.utils import degree

from Model import MyModel
from Utils import get_optim
from Augmentation import add_edges, add_edge_weights, drop_edges, drop_edge_weights, drop_feature_random, add_edge_random
from Linear_evaluation import linear_evaluation
from Load_data import get_dataset
from Arguments import get_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def tensor_mean(value):
    if value is None or value.numel() == 0:
        return 0.0
    return float(value.float().mean().item())


def edge_tuple_set(edge_index, max_edges=200000):
    if edge_index.size(1) > max_edges:
        return None
    edge_cpu = edge_index.detach().cpu()
    return set(zip(edge_cpu[0].tolist(), edge_cpu[1].tolist()))


def edge_view_stats(original_edge_index, edge_index_1, edge_index_2, deg):
    stats = {}
    original = edge_tuple_set(original_edge_index)
    if original is None:
        return stats
    original_count = max(1, len(original))
    for name, edge_index in [("view1", edge_index_1), ("view2", edge_index_2)]:
        view = edge_tuple_set(edge_index)
        if view is None:
            continue
        added = list(view - original)
        dropped = list(original - view)
        stats[f"{name}_added_edge_ratio"] = len(added) / original_count
        stats[f"{name}_dropped_edge_ratio"] = len(dropped) / original_count
        if added:
            add_src = torch.tensor([u for u, _ in added], device=deg.device)
            add_dst = torch.tensor([v for _, v in added], device=deg.device)
            stats[f"{name}_added_endpoint_degree"] = tensor_mean(torch.cat([deg[add_src], deg[add_dst]]))
        else:
            stats[f"{name}_added_endpoint_degree"] = 0.0
        if dropped:
            drop_src = torch.tensor([u for u, _ in dropped], device=deg.device)
            drop_dst = torch.tensor([v for _, v in dropped], device=deg.device)
            stats[f"{name}_dropped_endpoint_degree"] = tensor_mean(torch.cat([deg[drop_src], deg[drop_dst]]))
        else:
            stats[f"{name}_dropped_endpoint_degree"] = 0.0
    return stats


@torch.no_grad()
def augmentation_consistency(z1, z2):
    if z1 is None or z2 is None:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean().item())


@torch.no_grad()
def neighborhood_inconsistency(z, edge_index):
    if z is None or edge_index.numel() == 0:
        return 0.0
    z = torch.nn.functional.normalize(z, dim=1)
    src, dst = edge_index
    return float((1 - (z[src] * z[dst]).sum(dim=1)).mean().item())


@torch.no_grad()
def false_negative_proxy(x, edge_index, sample_size=1024, topk=10):
    num_nodes = x.size(0)
    if num_nodes <= 1:
        return 0.0
    sample_size = min(sample_size, num_nodes)
    sample = torch.arange(sample_size, device=x.device)
    if num_nodes > sample_size:
        sample = torch.randperm(num_nodes, device=x.device)[:sample_size]

    x_norm = torch.nn.functional.normalize(x.float(), dim=1)
    sim = x_norm[sample] @ x_norm.t()
    sim[torch.arange(sample_size, device=x.device), sample] = -1
    k = min(topk, num_nodes - 1)
    top_idx = torch.topk(sim, k=k, dim=1).indices.detach().cpu()

    edge_cpu = edge_index.detach().cpu()
    neighbors = {}
    for u, v in zip(edge_cpu[0].tolist(), edge_cpu[1].tolist()):
        neighbors.setdefault(u, set()).add(v)
    sample_cpu = sample.detach().cpu().tolist()
    likely_false_negative = 0
    total = 0
    for row, node in enumerate(sample_cpu):
        node_neighbors = neighbors.get(node, set())
        for candidate in top_idx[row].tolist():
            if candidate != node and candidate not in node_neighbors:
                likely_false_negative += 1
            total += 1
    return likely_false_negative / max(1, total)


def mean_numeric_dicts(dicts):
    values = {}
    for item in dicts:
        for key, value in item.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.setdefault(key, []).append(float(value))
    return {key: float(np.mean(value)) for key, value in values.items() if value}


def train(data, edge_weights, adding_edge, dropping_edge_weights, args, repeat_id):
    model = MyModel(data.num_features, args.feat_dim, args.proj_hidden_dim, args.temperature).to(args.device)
    optimizer = get_optim(model.parameters(), args)
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).to(args.device)

    best_ac = 0.0
    best_eval_metrics = {}
    last_z1, last_z2 = None, None
    last_edge_index_1, last_edge_index_2 = data.edge_index, data.edge_index

    train_bar = tqdm(range(args.epoch), desc='Training')
    for epoch in train_bar:
        model.train()
        optimizer.zero_grad()

        if args.drop_edge:
            e1 = drop_edges(data.edge_index, dropping_edge_weights, args.edge_drop_rate_1)
            e2 = drop_edges(data.edge_index, dropping_edge_weights, args.edge_drop_rate_2)
        else:
            e1 = data.edge_index
            e2 = data.edge_index

        if args.add_edge_random:
            edge_index_1 = add_edge_random(e1, args.edge_add_rate)
            edge_index_2 = add_edge_random(e2, args.edge_add_rate)
        elif args.add_edge:
            edge_index_1 = add_edges(edge_weights, adding_edge, e1)
            edge_index_2 = add_edges(edge_weights, adding_edge, e2)
        else:
            edge_index_1 = e1
            edge_index_2 = e2
        if args.add_single:
            edge_index_2 = e2

        if args.drop_feature_random:
            x_1 = drop_feature_random(data.x, args.feat_drop_rate_1)
            x_2 = drop_feature_random(data.x, args.feat_drop_rate_2)
        else:
            x_1 = data.x
            x_2 = data.x

        _, z1 = model(x_1, edge_index_1)
        _, z2 = model(x_2, edge_index_2)
        loss = model.loss(z1, z2, args.batch_compute)
        loss.backward()
        optimizer.step()

        last_z1, last_z2 = z1.detach(), z2.detach()
        last_edge_index_1, last_edge_index_2 = edge_index_1.detach(), edge_index_2.detach()
        train_bar.set_description("Train epoch {}, loss: {:.4f}".format(epoch + 1, loss.item()))

        if (epoch + 1) % args.loss_log == 0:
            path = args.output_path + "/loss.txt"
            with open(path, "a") as f:
                print("Train epoch {}, loss: {:.4f}".format(epoch + 1, loss.item()), file=f)
        if (epoch + 1) % args.eval == 0:
            path = args.output_path + "/eval.txt"
            model.eval()
            z = model(data.x, data.edge_index)[0].detach()
            y = data.y
            split_seed = args.seed + repeat_id if args.fixed_split else None
            output, ac, eval_metrics = linear_evaluation(z, y, args, epoch + 1, split_seed=split_seed, degree=deg)
            if ac > best_ac:
                best_ac = ac
                best_eval_metrics = eval_metrics
                torch.save(model.state_dict(), args.output_path + '/best_ac_ckpt.pth')
            with open(path, "a") as f:
                print(output, file=f)
        if (epoch + 1) % args.save_model == 0:
            path = args.output_path + "/ckpt_epoch{}.pth".format(epoch + 1)
            torch.save(model.state_dict(), path)
    train_bar.close()

    model.eval()
    z = model(data.x, data.edge_index)[0].detach()
    diagnostics = {
        "augmentation_consistency": augmentation_consistency(last_z1, last_z2),
        "neighbor_inconsistency": neighborhood_inconsistency(z, data.edge_index),
        "false_negative_proxy": false_negative_proxy(data.x, data.edge_index),
    }
    diagnostics.update(best_eval_metrics)
    diagnostics.update(edge_view_stats(data.edge_index, last_edge_index_1, last_edge_index_2, deg))
    print("best ac: {:.4f}".format(best_ac))
    return best_ac, diagnostics


def setup(data, args):
    if args.not_add_edge:
        adding_edge_weights, adding_edge = None, None
    else:
        adding_edge_weights, adding_edge = add_edge_weights(data.edge_index, args.edge_add_rate)
    if args.not_drop_edge:
        dropping_edge_weights = None
    else:
        dropping_edge_weights = drop_edge_weights(data.edge_index)
    return adding_edge_weights, adding_edge, dropping_edge_weights


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    data = get_dataset(args.dataset_path, args.dataset).to(args.device)
    t0 = time()
    adding_edge_weights, adding_edge, dropping_edge_weights = setup(data, args)
    t1 = time()
    print(f"Data Processing Time: {t1 - t0:.2f}s")
    output_path = args.output_path

    print(f"dataset:{args.dataset}  device:{args.device}  feature dim:{args.feat_dim}")
    acc = []
    diagnostics = []
    for i in range(args.repeat):
        set_seed(args.seed + i)
        args.output_path = output_path + '/' + str(i + 1)
        ac, diag = train(data, adding_edge_weights, adding_edge, dropping_edge_weights, args, i)
        acc.append(ac)
        diagnostics.append(diag)

    acc = np.array(acc)
    diag_summary = mean_numeric_dicts(diagnostics)
    result = {
        "dataset": args.dataset,
        "mean_acc": float(acc.mean()),
        "std_acc": float(acc.std()),
        "max_acc": float(acc.max()),
        "min_acc": float(acc.min()),
        **diag_summary,
    }
    print("max:{:.2f}, min:{:.2f}, mean:{:.2f}, std:{:.2f}".format(acc.max(), acc.min(), acc.mean(), acc.std()))
    with open(output_path + '/result.txt', "a") as f:
        print(acc, file=f)
        print("mean:{:.2f}, std:{:.2f}".format(acc.mean(), acc.std()), file=f)
        print("diagnostics:" + json.dumps(diag_summary, sort_keys=True), file=f)
    if args.emit_json:
        print("AUTORESEARCH_RESULT " + json.dumps(result, sort_keys=True))
    completed_dir = output_path.replace('in-progress', 'completed')
    os.rename(output_path, completed_dir)
    print("result saved to" + completed_dir)
