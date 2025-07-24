import torch


def remap_querymask(mask, query_labels, selected_classes):
    remap = torch.full_like(mask, fill_value=len(selected_classes)).long()
    for idx, cls in enumerate(query_labels):
        class_idx = selected_classes.index(cls)
        remap[idx][mask[idx] == 255] = class_idx
    return remap

def reverse_remap(pred, selected_classes):
    """
    将 pred (0 前景, 1 背景) 恢复成 255/0 二值掩码
    """
    bin_mask = torch.zeros_like(pred)
    for idx in range(len(selected_classes)):
        bin_mask[pred == idx] = 255
    return bin_mask