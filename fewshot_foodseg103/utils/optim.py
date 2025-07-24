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

    elif name == 'onecycle':
        # 注意：OneCycle 必须 batch 内 step 调用，且需要总步数和最大 lr
        max_lr = kwargs.pop('max_lr', 1e-2)
        total_steps = kwargs.pop('total_steps', 1000)
        pct_start = kwargs.pop('pct_start', 0.3)
        div_factor = kwargs.pop('div_factor', 25.0)
        final_div_factor = kwargs.pop('final_div_factor', 1e4)
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps,
            pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor
        )

    elif name == 'none':
        return None

    else:
        raise ValueError(f"Unsupported scheduler: {name}")
    

"""
- cosine_restart（CosineAnnealingWarmRestarts）：余弦退火调度，每隔 T_0 轮进行一次学习率周期性重启，可有效跳出局部最优；
- onecycle（OneCycleLR）：训练初期快速提升学习率，后期大幅下降，适合加速收敛，必须以 batch 为单位更新 step。
- 若使用 OneCycleLR，需要在 Trainer 中将 scheduler.step() 从每 epoch 改为每 batch 更新。
- 若使用 ReduceLROnPlateau，需确保传入 main_metric，并在 validation 后调用 scheduler.step(metric)。

get_scheduler(optimizer, name='onecycle', total_steps=1000, max_lr=1e-2)
get_scheduler(optimizer, name='cosine_restart', T_0=10, T_mult=2)

默认支持：
- step：固定周期衰减
- cosine：单周期余弦退火
- cosine_restart：周期性余弦重启
- plateau：根据验证集指标自适应调整
- onecycle：训练前期升后期降
- none：不使用调度器

拓展：
LinearWarmup、PolynomialDecay、ExponentialDecay

后续：
- 在 Trainer 中支持 batch-wise 的 scheduler.step() 调用（用于 onecycle）
- 与 wandb 日志联动记录当前学习率变化曲线
确保 train.py 中根据 scheduler 类型动态选择 epoch/batch 的 step 调用频率。

注意：
OneCycleLR 中的 total_steps 是必须指定的（= epoch * steps_per_epoch）。
CosineAnnealingWarmRestarts 的 T_0 是首次周期长度。
"""