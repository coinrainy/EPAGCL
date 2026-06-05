from torch.optim import Adam, SGD


def get_optim(params, args_or_optim, lr=None, weight_decay=None):
    if isinstance(args_or_optim, str):
        optim = args_or_optim
        lr = 0.01 if lr is None else lr
        weight_decay = 0.0 if weight_decay is None else weight_decay
        momentum = 0.9
    else:
        args = args_or_optim
        optim = args.optim
        lr = args.lr
        weight_decay = args.weight_decay
        momentum = getattr(args, "momentum", 0.9)

    if optim == 'Adam':
        return Adam(params=params, lr=lr, weight_decay=weight_decay)
    if optim == 'SGD':
        return SGD(params=params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optim}")
