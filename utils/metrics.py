import torch


class MeanIoU:
    def __init__(self, num_classes, smooth=1e-6, ignore_index=None, per_image=False):
        """
        num_classes: 总类别数
        smooth: 平滑项，避免除零
        ignore_index: 忽略的类别
        per_image: 是否对每张图单独计算 mIoU
        """

        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.per_image = per_image

    def __call__(self, preds, targets):
        # 预测类别和真实类别
        preds = torch.argmax(preds, dim=1)  # [B, H, W]
        targets = targets.to(preds.device)

        # 对整个batch计算 miou
        ious = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            pred_cls = (preds == cls).float()
            target_cls = (targets == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            if union == 0:
                # class not exist, default nan // Avoid empty class interference average
                iou = torch.tensor(float('nan'), device=preds.device)
            else:
                iou = (intersection + self.smooth) / (union + self.smooth)
            ious.append(iou)
        batch_miou = torch.nanmean(torch.stack(ious))

        if not self.per_image:
            return batch_miou
        
        # 若每张图单独计算 mIoU
        per_image_miou = []
        for b in range(preds.shape[0]):
            ious = []
            for cls in range(self.num_classes):
                if cls == self.ignore_index:
                    continue

                pred_cls = (preds[b] == cls).float()
                target_cls = (targets[b] == cls).float()
                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum() - intersection
                if union == 0:
                    iou = torch.tensor(float('nan'), device=preds.device)
                else:
                    iou = (intersection + self.smooth) / (union + self.smooth)
                ious.append(iou)
            per_image_miou.append(torch.nanmean(torch.stack(ious)))

        return {
            'batch_miou': batch_miou,
            'per_image_miou': torch.stack(per_image_miou)  # shape: [B]
        }
    
    def __repr__(self):
        return (f"<MeanIoU(num_classes={self.num_classes}, "
                f"smooth={self.smooth}, "
                f"ignore_index={self.ignore_index}, "
                f"per_image={self.per_image})>")


class DiceScore:
    def __init__(self, num_classes, smooth=1e-6, ignore_index=None, per_image=False):
        """
        num_classes: 总类别数
        smooth: 平滑项，避免除零
        ignore_index: 忽略的类别
        per_image: 是否对每张图单独计算 mdice
        """

        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.per_image = per_image

    def __call__(self, preds, targets):
        preds = torch.argmax(preds, dim=1)  # [B, H, W]
        targets = targets.to(preds.device)

        # 对整个 batch 计算 mdice
        dices = []
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            pred_cls = (preds == cls).float()
            target_cls = (targets == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            if union == 0:
                dice = torch.tensor(float('nan'), device=preds.device)
            else:
                dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dices.append(dice)
        batch_mdice = torch.nanmean(torch.stack(dices))

        if not self.per_image:
            return batch_mdice

        # 若每张图单独计算 mdice
        per_image_mdice = []
        for b in range(preds.shape[0]):
            dices = []
            for cls in range(self.num_classes):
                if cls == self.ignore_index:
                    continue
                pred_cls = (preds[b] == cls).float()
                target_cls = (targets[b] == cls).float()
                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum()
                if union == 0:
                    dice = torch.tensor(float('nan'), device=preds.device)
                else:
                    dice = (2 * intersection + self.smooth) / (union + self.smooth)
                dices.append(dice)
            per_image_mdice.append(torch.nanmean(torch.stack(dices)))

        return {
            'batch_mdice': batch_mdice,
            'per_image_mdice': torch.stack(per_image_mdice)
        }

    def __repr__(self):
        return (f"<DiceScore(num_classes={self.num_classes}, "
                f"smooth={self.smooth}, "
                f"ignore_index={self.ignore_index}, "
                f"per_image={self.per_image})>")
    

class PixelAccuracy:
    def __init__(self, ignore_index=None, per_image=False):
        """
        ignore_index: 忽略的类别（像素值）
        per_image: 是否对每张图单独计算 Accuracy
        """

        self.ignore_index = ignore_index
        self.per_image = per_image

    def __call__(self, preds, targets):
        preds = torch.argmax(preds, dim=1)  # [B, H, W]
        targets = targets.to(preds.device)

        # 整个 batch 计算 Accuracy
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            correct = ((preds == targets) & valid_mask).float().sum()
            total = valid_mask.float().sum()
        else:
            correct = (preds == targets).float().sum()
            total = preds.numel()

        batch_acc = correct / total if total > 0 else torch.tensor(0.0, device=preds.device)

        if not self.per_image:
            return batch_acc

        # 每张图单独计算 Accuracy
        per_image_acc = []
        for b in range(preds.shape[0]):
            pred_b = preds[b]
            target_b = targets[b]
            if self.ignore_index is not None:
                mask = (target_b != self.ignore_index)
                correct = ((pred_b == target_b) & mask).float().sum()
                total = mask.float().sum()
            else:
                correct = (pred_b == target_b).float().sum()
                total = pred_b.numel()
            acc = correct / total if total > 0 else torch.tensor(0.0, device=preds.device)
            per_image_acc.append(acc)

        return {
            'batch_acc': batch_acc,
            'per_image_acc': torch.stack(per_image_acc)
        }

    def __repr__(self):
        return (f"<PixelAccuracy(ignore_index={self.ignore_index}, "
                f"per_image={self.per_image})>")


class MetricCollection:
    def __init__(self, metrics: dict):
        """
        metrics: dict[str, metric_object]
        每个 metric_object 返回标量或 dict
        """

        self.metrics = metrics

    def __call__(self, preds, targets):
        results = {}
        for name, metric in self.metrics.items():
            out = metric(preds, targets)
            if isinstance(out, dict):
                # 拆分每个子结果，加前缀
                for k, v in out.items():
                    results[f"{name}/{k}"] = v
            else:
                results[name] = out
        return results

    def keys(self):
        return list(self.metrics.keys())

    def __repr__(self):
        return f"<MetricCollection(metrics={list(self.metrics.keys())})>"


def get_metric(names=['miou', 'dice', 'acc'], num_classes=None, prefix='', 
               ignore_index=None, per_image=False, **kwargs):
    """
    names: 要使用的指标名称列表
    num_classes: 类别数，用于 miou 和 dice
    prefix: 每个 metric 名字前缀（如 'val' → 'val/miou'）
    ignore_index: 忽略的类别标签
    per_image: 是否返回每张图的指标
    kwargs: 额外传给各个 metric 的参数
    """

    if isinstance(names, str):
        names = [names]

    available = {
        'miou': MeanIoU(num_classes=num_classes, ignore_index=ignore_index, per_image=per_image, **kwargs),
        'dice': DiceScore(num_classes=num_classes, ignore_index=ignore_index, per_image=per_image, **kwargs),
        'acc': PixelAccuracy(ignore_index=ignore_index, per_image=per_image)
    }

    selected = {}
    for name in names:
        name = name.lower()
        if name not in available:
            raise ValueError(f"Unsupported metric: {name}")
        key = f"{prefix}/{name}" if prefix else name
        selected[key] = available[name]

    return MetricCollection(selected)