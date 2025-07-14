# coding: utf-8
"""
Prototype‑ResNet50 with explicit **background channel**
======================================================
* n_way foreground classes + **1 background** channel (index = n_way).
* Background logits produced by a learnable scalar bias β (per-pixel constant).
* Loss/metrics should use `CrossEntropyLoss(ignore_index=None)` and remap
  non‑episode pixels to label `n_way`.
* Forward signature unchanged: `forward(support_set, query_set, selected_classes)`.
"""
from typing import List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models

Tensor = torch.Tensor

def l2norm(x: Tensor, dim: int = 1, eps: float = 1e-7):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def resnet50_stride8(pretrained=True):
    net = models.resnet50(pretrained=pretrained,
                          replace_stride_with_dilation=(False, True, True))
    layers = [
        nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool),
        net.layer1,  # 1/4
        net.layer2,  # 1/8
        net.layer3,  # 1/8 (dilated)
        net.layer4,  # 1/8
    ]
    return nn.Sequential(*layers)


class ProtoResNet50BG(nn.Module):
    def __init__(self, n_way: int, pretrained_backbone: bool = True):
        super().__init__()
        self.n_way = n_way              # foreground count
        self.n_out = n_way + 1          # +1 background

        self.encoder = resnet50_stride8(pretrained_backbone)
        self.feat_dim = 2048            # layer4 channels

        # lightweight head (1×1 conv + GN + ReLU)
        self.head = nn.Sequential(
            nn.Conv2d(self.n_out, self.n_out, 1, bias=False),
            nn.GroupNorm(num_groups=min(32, self.n_out), num_channels=self.n_out),
            nn.ReLU(inplace=True),
        )

        # learnable background bias β (scalar)
        self.bg_bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def _feat(self, x: Tensor):
        f = self.encoder(x)             # B,C,h,w  h=w=H/8
        return l2norm(f, 1)

    # ------------------------------------------------------------------
    def _prototypes(self, feats: Tensor, masks: Tensor, labels: Tensor, classes):
        """Return tensor (n_way, C) foreground prototypes."""
        C, H, W = feats.shape[1:]
        protos = []
        for cls_id in classes:
            idx = (labels == cls_id).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:                       # fallback zero
                protos.append(torch.zeros(C, device=feats.device))
                continue
            f = feats[idx]        # S',C,H,W
            m = masks[idx]        # S',H_org,W_org
            m = (m == cls_id).float()
            m = F.interpolate(m.unsqueeze(1), (H, W), mode='nearest').squeeze(1)
            area = m.sum((1,2))
            per = (f * m.unsqueeze(1)).sum((2,3)) / (area.unsqueeze(1) + 1e-7)
            protos.append(per.mean(0))
        return l2norm(torch.stack(protos), 1)          # (n_way,C)

    # ------------------------------------------------------------------
    def forward(self,
                support_set: List[Tuple[Tensor, Tensor, int]],
                query_set:   List[Tuple[Tensor, Tensor]],
                selected_classes: List[int]):
        # stack
        s_imgs  = torch.stack([i for i,_,_ in support_set])      # (S,3,H,W)
        s_masks = torch.stack([m for _,m,_ in support_set])      # (S,H,W)
        s_lbl   = torch.tensor([c for _,_,c in support_set], device=s_imgs.device)
        q_imgs  = torch.stack([i for i,_ in query_set])          # (Q,3,H,W)

        # feats
        s_feat = self._feat(s_imgs)   # S,C,h,w
        q_feat = self._feat(q_imgs)   # Q,C,h,w

        # prototypes & similarity
        fg_proto = self._prototypes(s_feat, s_masks, s_lbl, selected_classes)  # (n_way,C)
        sims     = torch.einsum('bchw,nc->bnhw', q_feat, fg_proto)             # Q,n_way,h,w

        # append background bias channel
        bg = self.bg_bias.expand(q_feat.size(0), 1, sims.size(2), sims.size(3))
        sims = torch.cat([sims, bg], dim=1)        # (Q,n_way+1,h,w)

        logits_low  = self.head(sims)
        logits_high = F.interpolate(logits_low, scale_factor=8, mode='bilinear', align_corners=False)
        return logits_high                         # (Q,n_out,H,W)


# quick smoke‑test
if __name__ == "__main__":
    model = ProtoResNet50BG(n_way=5, pretrained_backbone=False)
    sup = [(torch.randn(3,256,256), torch.randint(0,20,(256,256)), 7) for _ in range(5)]
    que = [(torch.randn(3,256,256), torch.randint(0,20,(256,256))) for _ in range(2)]
    out = model(sup, que, [7,11,4,3,17])
    print(out.shape)  # (2,6,256,256)
