import torch


def remap_querymask(mask, selected_classes, n_way):
    """
    将 query_mask 中原始类别ID映射为索引
    selected_classes[i] -> i (对齐输出的argmax顺序)
    其余类别 -> ignore_index
    """
    remapped = torch.full_like(mask, fill_value=n_way)
    for idx, cls_id in enumerate(selected_classes):
        remapped[mask == cls_id] = idx
    return remapped

def reverse_remap(pred_mask, selected_classes):
    """
    将 pred 中类别索引还原为原始类别ID
    """
    out = pred_mask.clone()
    for idx, cls_id in enumerate(selected_classes):
        out[pred_mask == idx] = cls_id
    return out