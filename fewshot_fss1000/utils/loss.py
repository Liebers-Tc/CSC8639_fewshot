import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class DiceLoss(nn.Module):
    def __init__(self, from_logits=True, weight=None):
        super().__init__()
        self.loss = smp.losses.DiceLoss(mode='multiclass', from_logits=from_logits, 
                                        classes=None if weight is None else torch.where(weight>0)[0].tolist())

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        if weight is not None:
            print("[Warn] smp.FocalLoss 不接受 weight ，忽略该参数")
        self.loss = smp.losses.FocalLoss(mode='multiclass', alpha=alpha, gamma=gamma)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class ComboLoss(nn.Module):
    """Dice + CrossEntropy """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, from_logits=True, weight=None):
        super().__init__()
        self.dice = DiceLoss(from_logits=from_logits, weight=weight)
        self.ce = CrossEntropyLoss(weight=weight)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, y_pred, y_true):
        return self.dice_weight * self.dice(y_pred, y_true) + \
               self.ce_weight * self.ce(y_pred, y_true)


def get_loss(name='ce', weight=None, **kwargs):
    name = name.lower()
    if name == 'ce':
        return CrossEntropyLoss(weight=weight)
    elif name == 'dice':
        return DiceLoss(weight=weight, **kwargs)
    elif name == 'focal':
        return FocalLoss(weight=weight, **kwargs)
    elif name == 'combo':
        return ComboLoss(weight=weight, **kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {name}")