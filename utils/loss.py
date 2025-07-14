import torch.nn as nn
import segmentation_models_pytorch as smp


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class DiceLoss(nn.Module):
    def __init__(self, from_logits=True, ignore_index=None):
        super().__init__()
        self.loss = smp.losses.DiceLoss(mode='multiclass', from_logits=from_logits)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=None):
        super().__init__()
        self.loss = smp.losses.FocalLoss(mode='multiclass', alpha=alpha, gamma=gamma)

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)


class ComboLoss(nn.Module):
    """Dice + CrossEntropy """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, from_logits=True, ignore_index=None):
        super().__init__()
        self.dice = DiceLoss(from_logits=from_logits)
        self.ce = CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, y_pred, y_true):
        return self.dice_weight * self.dice(y_pred, y_true) + \
               self.ce_weight * self.ce(y_pred, y_true)


def get_loss(name='ce', ignore_index=None, **kwargs):
    name = name.lower()
    if name == 'ce':
        return CrossEntropyLoss(ignore_index=ignore_index)
    elif name == 'dice':
        return DiceLoss(ignore_index=ignore_index, **kwargs)
    elif name == 'focal':
        return FocalLoss(ignore_index=ignore_index, **kwargs)
    elif name == 'combo':
        return ComboLoss(ignore_index=ignore_index, **kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {name}")