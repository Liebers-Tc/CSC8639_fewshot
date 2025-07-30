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
# Euclidean distance: ||q - p||² = q² + p² - 2q·p
def euclidean_dist(q, p):  # q: (B,HW,C), p: (n,C)
    q2 = (q ** 2).sum(dim=2, keepdim=True)  # (B,HW,1)
    p2 = (p ** 2).sum(dim=1).view(1, 1, -1)  # (1,1,n)
    qp = torch.matmul(q, p.T)  # (B,HW,n)
    return q2 + p2 - 2 * qp  # (B,HW,n)

# proto_compute
# --------------------------------------------------
def compute_prototypes(feats, masks, labels, classes):
    """
    feats : (S,C,h,w)   support image 特征
    masks : (S,H,W)     原始掩码
    labels: (S,)        对应类别
    classes: episode 选中的类别列表
    """

    H, W = feats.shape[2:]
    prototypes = []
    for cls_id in classes:
        idx = (labels == cls_id).nonzero(as_tuple=True)[0]
        f = feats[idx]
        m = masks[idx]

        m = (m == 255).float()
        m = F.interpolate(m.unsqueeze(1), (H, W), mode='nearest').squeeze(1)
        area = m.sum((1,2))
        per = (f * m.unsqueeze(1)).sum((2,3)) / (area.unsqueeze(1) + 1e-7)
        prototypes.append(per.mean(0))

    return torch.stack(prototypes)

# ProtoNet
# --------------------------------------------------
class ProtoNet(nn.Module):
    def __init__(self, n_way):
        super().__init__()
        self.n_way = n_way
        self.encoder, feat_dim, _ = encoder()
        self.decoder = decoder(n_way + 1)
        # 全局先验背景原型
        self.gl_bg_proto = nn.Parameter(torch.empty(1, feat_dim))  # (1, 2048)
        nn.init.normal_(self.gl_bg_proto, mean=0.0, std=0.02)

    def forward(self, support_set, query_set, selected_classes):
        # stack tensors
        s_imgs = torch.stack([img for img, _, _ in support_set])
        s_masks = torch.stack([mask for _, mask, _ in support_set])
        s_labels = torch.tensor([label for _, _, label in support_set], device=s_imgs.device)
        q_imgs = torch.stack([img for img, _, _ in query_set])

        # feature extraction
        s_feat = self.encoder(s_imgs)
        q_feat = self.encoder(q_imgs)

        # prototypes
        fg_proto = compute_prototypes(s_feat, s_masks, s_labels, selected_classes)
        bg_proto = self.gl_bg_proto
        proto = torch.cat([fg_proto, bg_proto], dim=0)
        
        B, C, H, W = q_feat.shape
        q_feat_flat = q_feat.view(B, C, -1).permute(0, 2, 1)

        dist = euclidean_dist(q_feat_flat, proto)
        sim = -dist.permute(0, 2, 1).view(B, self.n_way + 1, H, W)  # 负距离作为相似度

        logits = self.decoder(sim)
        return logits
