import torch


class MeanIoU:
    def __init__(self, smooth=1e-6, per_image=False):
        """
        smooth: 平滑项，避免除零
        per_image: 是否对每张图单独计算 mIoU
        """
        self.smooth = smooth
        self.per_image = per_image

    def __call__(self, preds, targets, selected_classes):
        # 对整个batch计算 miou
        ious = []
        for cls in selected_classes:
            pred_cls = (preds == cls).float()
            target_cls = (targets == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            if union == 0:
                continue  # 跳过未出现的类
            iou = (intersection + self.smooth) / (union + self.smooth)
            ious.append(iou)
        batch_miou = torch.nanmean(torch.stack(ious)) if ious else torch.tensor(0.0, device=preds.device)

        if not self.per_image:
            return batch_miou
        
        # 若每张图单独计算 mIoU
        per_image_miou = []
        for b in range(preds.shape[0]):
            ious = []
            for cls in selected_classes:
                pred_cls = (preds[b] == cls).float()
                target_cls = (targets[b] == cls).float()
                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum() - intersection
                if union == 0:
                    continue  # 跳过未出现的类
                iou = (intersection + self.smooth) / (union + self.smooth)
                ious.append(iou)
            per_img_iou = torch.nanmean(torch.stack(ious)) if ious else torch.tensor(0.0, device=preds.device)
            per_image_miou.append(per_img_iou)

        return {
            'batch_miou': batch_miou,
            'per_image_miou': torch.stack(per_image_miou)  # shape: [B]
        }


class DiceScore:
    def __init__(self, smooth=1e-6, per_image=False):
        """
        smooth: 平滑项，避免除零
        per_image: 是否对每张图单独计算 mdice
        """
        self.smooth = smooth
        self.per_image = per_image

    def __call__(self, preds, targets, selected_classes):
        # 对整个 batch 计算 mdice
        dices = []
        for cls in selected_classes:
            pred_cls = (preds == cls).float()
            target_cls = (targets == cls).float()
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            if union == 0:
                continue  # 跳过未出现的类
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dices.append(dice)
        batch_mdice = torch.nanmean(torch.stack(dices)) if dices else torch.tensor(0.0, device=preds.device)

        if not self.per_image:
            return batch_mdice

        # 若每张图单独计算 mdice
        per_image_mdice = []
        for b in range(preds.shape[0]):
            dices = []
            for cls in selected_classes:
                pred_cls = (preds[b] == cls).float()
                target_cls = (targets[b] == cls).float()
                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum()
                if union == 0:
                    continue
                dice = (2 * intersection + self.smooth) / (union + self.smooth)
                dices.append(dice)
            per_img_dice = torch.nanmean(torch.stack(dices)) if dices else torch.tensor(0.0, device=preds.device)
            per_image_mdice.append(per_img_dice)

        return {
            'batch_mdice': batch_mdice,
            'per_image_mdice': torch.stack(per_image_mdice)
        }
    

class PixelAccuracy:
    def __init__(self, per_image=False):
        """
        per_image: 是否对每张图单独计算 Accuracy
        """
        self.per_image = per_image

    def __call__(self, preds, targets, selected_classes):
        # 整个 batch 计算 Accuracy
        mask = torch.zeros_like(targets, dtype=torch.bool)  # 初始化全False
        for cls in selected_classes:
            mask |= (targets == cls)

        correct = ((preds == targets) & mask).float().sum()
        total = mask.float().sum()
        batch_acc = correct / total if total > 0 else torch.tensor(0.0, device=preds.device)

        if not self.per_image:
            return batch_acc

        # 每张图单独计算 Accuracy
        per_image_acc = []
        for b in range(preds.shape[0]):
            mask_b = torch.zeros_like(targets[b], dtype=torch.bool)
            for cls in selected_classes:
                mask_b |= (targets[b] == cls)

            correct_b = ((preds[b] == targets[b]) & mask_b).float().sum()
            total_b = mask_b.float().sum()
            acc = correct_b / total_b if total_b > 0 else torch.tensor(0.0, device=preds.device)
            per_image_acc.append(acc)

        return {
            'batch_acc': batch_acc,
            'per_image_acc': torch.stack(per_image_acc)
        }
    

class Precision:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, preds, targets, selected_classes):
        precisions = []
        for cls in selected_classes:
            tp = ((preds == cls) & (targets == cls)).float().sum()
            fp = ((preds == cls) & (targets != cls)).float().sum()
            precisions.append(tp / (tp + fp + self.eps))
        return torch.nanmean(torch.stack(precisions))
    

class Recall:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, preds, targets, selected_classes):
        recalls = []
        for cls in selected_classes:
            tp = ((preds == cls) & (targets == cls)).float().sum()
            fn = ((preds != cls) & (targets == cls)).float().sum()
            recalls.append(tp / (tp + fn + self.eps))
        return torch.nanmean(torch.stack(recalls))
    

# class ConfusionMatrix:
#     def __call__(self, preds, targets, selected_classes):
#         num_cls = len(selected_classes) + 1
#         k = (targets >= 0) & (targets < num_cls)
#         return torch.bincount(num_cls * targets[k] + preds[k], minlength=num_cls**2).reshape(num_cls, num_cls)



class MetricCollection:
    def __init__(self, metrics):
        """
        metrics: dict[str, metric_object]
        每个 metric_object 返回标量或 dict
        """
        self.metrics = metrics

    def __call__(self, preds, targets, selected_classes):
        results = {}
        for name, metric in self.metrics.items():
            out = metric(preds, targets, selected_classes)
            if isinstance(out, dict):
                # 拆分每个子结果，加前缀
                for k, v in out.items():
                    results[f"{name}/{k}"] = v
            else:
                results[name] = out
        return results

    def keys(self):
        return list(self.metrics.keys())


def get_metric(names=['miou', 'dice', 'acc'], prefix='', per_image=False):
    """
    names: 要使用的指标名称列表
    prefix: 每个 metric 名字前缀（如 'val' → 'val/miou'）
    per_image: 是否返回每张图的指标
    kwargs: 额外传给各个 metric 的参数
    """
    if isinstance(names, str):
        names = [names]

    available = {
        'miou': MeanIoU(per_image=per_image),
        'dice': DiceScore(per_image=per_image),
        'acc': PixelAccuracy(per_image=per_image),
        'precision': Precision(),
        'recall': Recall(),
        # 'cm': ConfusionMatrix(),
    }

    selected = {}
    for name in names:
        name = name.lower()
        if name not in available:
            raise ValueError(f"Unsupported metric: {name}")
        key = f"{prefix}/{name}" if prefix else name
        selected[key] = available[name]

    return MetricCollection(selected)