import torch

from tqdm import tqdm
from torch.utils.data import random_split

from Model import LinearClassifier
from Utils import get_optim


def linear_evaluation(z, y, args, epochs, split=None, split_seed=None, degree=None):
    classes_num = y.max().item() + 1
    y = y.view(-1)
    feat_dim = z.size(1)

    if split is None:
        train_mask, test_mask, val_mask = dataset_split(z.size(0), seed=split_seed, device=z.device)
    else:
        train_mask, test_mask, val_mask = split['train'], split['test'], split['valid']

    train_set, train_label = z[train_mask], y[train_mask]
    test_set, test_label = z[test_mask], y[test_mask]
    val_set, val_label = z[val_mask], y[val_mask]

    classifier = LinearClassifier(feat_dim, classes_num).to(args.device)
    optimizer = get_optim(classifier.parameters(), args.eval_optim, args.eval_lr, args.eval_weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_test_acc = 0
    best_epoch = 0
    best_eval_path = args.output_path + '/best_eval.pth'

    eval_bar = tqdm(range(args.eval_epoch), desc='Evaluating')
    for epoch in eval_bar:
        classifier.train()
        optimizer.zero_grad()

        output = classifier(train_set)
        loss = criterion(output, train_label)
        loss.backward()
        optimizer.step()

        classifier.eval()
        val_acc = ac(classifier, val_set, val_label)
        test_acc = ac(classifier, test_set, test_label)
        if val_acc > best_val_acc or (val_acc == best_val_acc and test_acc > best_test_acc):
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save(classifier.state_dict(), best_eval_path)

        eval_bar.set_description("Eval epoch {}, best test acc: {:.4f}".format(epochs, best_test_acc))
        eval_bar.set_postfix(loss=loss.item())
    eval_bar.close()
    classifier.load_state_dict(torch.load(best_eval_path, weights_only=True))
    classifier.eval()

    metrics = {}
    if degree is not None:
        metrics.update(degree_bucket_accuracy(classifier, test_set, test_label, degree[test_mask]))
    return "best epoch: {}, best test acc: {:.4f}".format(best_epoch, best_test_acc), best_test_acc, metrics


def ac(model, dataset, label):
    output = model(dataset)
    _, pred = torch.max(output, 1)
    total = label.size(0)
    correct = (pred == label).sum().item()
    return 100 * correct / total


def dataset_split(num, train_ratio=0.1, val_ratio=0.1, seed=None, device=None):
    train_len = int(num * train_ratio)
    val_len = int(num * val_ratio)
    test_len = num - train_len - val_len

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    train, test, val = random_split(torch.arange(0, num), (train_len, test_len, val_len), generator=generator)
    train_idx, test_idx, val_idx = train.indices, test.indices, val.indices
    return get_mask(num, train_idx, device), get_mask(num, test_idx, device), get_mask(num, val_idx, device)


def get_mask(length, idx, device=None):
    mask = torch.zeros(length, dtype=torch.bool)
    mask[idx] = True
    if device is not None:
        mask = mask.to(device)
    return mask


def degree_bucket_accuracy(model, dataset, label, degree):
    output = model(dataset)
    _, pred = torch.max(output, 1)
    metrics = {}
    if degree.numel() == 0:
        return metrics
    low_q = torch.quantile(degree.float(), 1 / 3)
    high_q = torch.quantile(degree.float(), 2 / 3)
    buckets = {
        "low_degree_acc": degree <= low_q,
        "mid_degree_acc": (degree > low_q) & (degree < high_q),
        "high_degree_acc": degree >= high_q,
    }
    for name, mask in buckets.items():
        if int(mask.sum()) == 0:
            metrics[name] = 0.0
        else:
            metrics[name] = 100 * (pred[mask] == label[mask]).sum().item() / int(mask.sum())
    return metrics
