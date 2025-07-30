import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# encoder
# --------------------------------------------------
def encoder(pretrained=True):
    """ResNet50 截至 layer4，输出 (B,2048,H/8,W/8)"""
    weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    net = resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])  # layer3/4 stride=1, dilation=2 → 总下采样 8×
    enc = nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool,  # 1/4
        net.layer1,                                 # 1/4
        net.layer2,                                 # 1/8 (C=512)
        net.layer3,                                 # 1/8 (dilated, C=1024)
        net.layer4                                  # 1/8 (dilated, C=2048)
    )
    return enc, 2048, 8

# decoder
# --------------------------------------------------
def decoder(channels):
    """3× /8→/4→/2→/1"""
    return nn.Sequential(
        nn.ConvTranspose2d(channels, 64, 4, stride=2, padding=1, bias=False),
        nn.GroupNorm(num_groups=4, num_channels=64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
        nn.GroupNorm(num_groups=2, num_channels=32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1, bias=False)
    )

# utilities
# --------------------------------------------------
def l2_normalize(x, eps=1e-7):
    return F.normalize(x, p=2, dim=1, eps=eps)

# proto_compute
# --------------------------------------------------
def compute_fg_prototypes(feats, masks, labels, classes):
    """
    feats: 支持集特征  (S,C,h,w)
    masks: 支持集原始掩码  (S,H,W)
    labels: 对应类别
    classes: 选中的类别列表
    """
    H, W = feats.shape[2:]
    fg_prototypes = []
    for cls_id in classes:
        idx = (labels == cls_id).nonzero(as_tuple=True)[0]
        f = feats[idx]
        m = masks[idx]

        m = (m == 255).float()
        m = F.interpolate(m.unsqueeze(1), (H, W), mode='nearest').squeeze(1)
        area = m.sum((1,2))
        per = (f * m.unsqueeze(1)).sum((2,3)) / (area.unsqueeze(1) + 1e-7)
        fg_prototypes.append(per.mean(0))
    return l2_normalize(torch.stack(fg_prototypes))  # (N_way, 2048)

def compute_bg_prototypes(feats, masks, labels, classes, merge_all=True):
    """
    计算单次 episode 的背景原型
    feats: 支持集特征  (S,C,h,w)
    masks: 支持集原始掩码  (S,H,W)
    classes: 选中的类别列表，用于反选背景
    max_pix: 最大选取像素数量
    """
    H, W = feats.shape[2:]
    bg_prototypes_per_class = {}
    for cls_id in classes:
        idx = (labels == cls_id).nonzero(as_tuple=True)[0]
        f = feats[idx]
        m = masks[idx]

        m = (m == 0).float()
        m = F.interpolate(m.unsqueeze(1), (H, W), mode='nearest').squeeze(1)
        area = m.sum((1,2))
        per = (f * m.unsqueeze(1)).sum((2,3)) / (area.unsqueeze(1) + 1e-7)
        bg_prototypes_per_class[cls_id] = per.mean(0)  # 每类平均（后续可以修改为每张图先建立一个原型，按图处理）

    if merge_all:
        bg_prototype = torch.stack(list(bg_prototypes_per_class.values())).mean(0, keepdim=True)
        return l2_normalize(bg_prototype)  # 统一一个背景原型
    else:  # 预留接口
        return {k: l2_normalize(v.unsqueeze(0)).squeeze(0) for k, v in bg_prototypes_per_class.items()}  # 每类一个背景原型

# ProtoNet
# --------------------------------------------------
class ProtoNet(nn.Module):
    def __init__(self, n_way, beta_mix, alpha_bg):
        """
        beta_mix: 全局先验背景原型与单次 episode 背景原型的融合比例
        0->只用全局先验背景原型；1->只用单次 episode 背景原型；中间值线性融合
        alpha_bg: 缩放背景分数，越小→背景影响越弱
        """
        super().__init__()
        self.n_way = n_way
        self.encoder, feat_dim, _ = encoder()
        self.decoder = decoder(n_way + 1)
        self.alpha_bg = alpha_bg
        self.beta_mix = beta_mix
        # 全局先验背景原型
        self.register_buffer("gl_bg_proto", torch.zeros(1, feat_dim))
        # self.gl_bg_proto = torch.zeros(1, feat_dim, device='cuda')
        
    def forward(self, support_set, query_set, selected_classes):
        s_imgs = torch.stack([img for img, _, _ in support_set])
        s_masks = torch.stack([mask for _, mask, _ in support_set])
        s_labels = torch.tensor([label for _, _, label in support_set], device=s_imgs.device)
        q_imgs = torch.stack([img for img, _, _ in query_set])

        s_feat = l2_normalize(self.encoder(s_imgs))
        q_feat = l2_normalize(self.encoder(q_imgs))

        # 前景原型
        fg_proto = compute_fg_prototypes(s_feat, s_masks, s_labels, selected_classes)
        
        # 背景原型
        ep_bg_proto = compute_bg_prototypes(s_feat, s_masks, s_labels, selected_classes)

        if self.beta_mix < 1:
            with torch.no_grad():
                self.gl_bg_proto.add_(ep_bg_proto)
                self.gl_bg_proto.copy_(l2_normalize(self.gl_bg_proto))

        bg_proto = self.beta_mix * ep_bg_proto + (1 - self.beta_mix) * self.gl_bg_proto
        bg_proto = l2_normalize(bg_proto)

        fg_sim = torch.einsum('bchw,nc->bnhw', q_feat, fg_proto)     
        bg_sim = torch.einsum('bchw,nc->bnhw',  q_feat, bg_proto)
        bg_sim = self.alpha_bg * bg_sim
        sim = torch.cat([fg_sim, bg_sim], dim=1)

        # Upsample
        logit = self.decoder(sim)
        return logit
