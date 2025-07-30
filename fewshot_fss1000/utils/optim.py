from torch import optim


def get_optimizer(model, name='adamw', lr=1e-3, weight_decay=1e-4, use_param_groups=False, **kwargs):
    """
    构建优化器，可选参数分组方式（为 bias 和 norm 层禁用 weight decay）
    model: 要训练的模型
    name (str): 优化器名称，支持 'adam'、'adamw'、'sgd'
    lr (float): 学习率
    weight_decay (float): 权重衰减（L2 正则化）
    use_param_groups (bool): 是否为不同类型参数设置不同的权重衰减
    kwargs: 传递给优化器的其他参数（如 betas, eps 等）
    """
    name = name.lower()
    if use_param_groups:
        # 参数分组：为 norm 层和 bias 设置 0 权重衰减，其余正常使用 weight_decay
        decay, no_decay = [], []
        for pname, param in model.named_parameters():
            if not param.requires_grad:
                continue  # 跳过冻结参数
            if 'bias' in pname or 'norm' in pname.lower():
                no_decay.append(param)  # 不加正则
            else:
                decay.append(param)     # 加正则

        # 每组参数单独指定是否加正则
        params = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]

    else:
        # 所有参数统一使用 weight_decay，如果 kwargs 中未指定，则自动添加
        params = model.parameters()
        kwargs.setdefault('weight_decay', weight_decay)

    # 根据优化器名称构建对应优化器实例
    if name == 'adam':
        return optim.Adam(params, lr=lr, **kwargs)
    elif name == 'adamw':
        return optim.AdamW(params, lr=lr, **kwargs)
    elif name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
    

def get_scheduler(optimizer, name='step', **kwargs):
    """
    optimizer: 优化器实例
    name (str): 调度器名称，支持 'step'、'cosine'、'plateau'、'cosine_restart'、'onecycle'、'none'
    kwargs: 调度器参数
    """
    name = name.lower()
    if name == 'step':
        step_size = kwargs.pop('step_size', 10)
        gamma = kwargs.pop('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif name == 'cosine':
        T_max = kwargs.pop('T_max', 10)
        eta_min = kwargs.pop('eta_min', 1e-7)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif name == 'cosine_restart':
        # CosineAnnealingWarmRestarts 支持余弦周期重启
        T_0 = kwargs.pop('T_0', 10)  # 首次重启的周期
        T_mult = kwargs.pop('T_mult', 2)  # 每次重启周期乘以的倍数
        eta_min = kwargs.pop('eta_min', 1e-7)
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )

    elif name == 'plateau':
        main_metric = kwargs.pop('main_metric')
        mode = kwargs.pop('mode', 'min' if 'loss' in main_metric.lower() else 'max')
        patience = kwargs.pop('patience', 6)
        factor = kwargs.pop('factor', 0.1)
        threshold = kwargs.pop('threshold', 1e-5)
        min_lr = kwargs.pop('min_lr', 1e-7)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=patience, factor=factor,
            threshold=threshold, min_lr=min_lr
        )

    else:
        raise ValueError(f"Unsupported scheduler: {name}")