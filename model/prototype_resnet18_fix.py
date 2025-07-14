# coding: utf-8
"""
PrototypeNet — stride‑8 backbone, no skip, no refine
===================================================
- **Backbone**: ResNet‑18 up to *layer2*  → output 1/8 (32×32) resolution,
  channel dim 128.
- **Decoder**: three UpBlocks (ConvT 4×4 s2 p1 + GroupNorm + ReLU) bringing
  32×32 → 64×64 → 128×128 → **256×256**.
- **No skip‑connections**, **no extra refine** layer.
- Forward / dataset interface与旧版保持一致。
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# helper
# -----------------------------------------------------------------------------

def l2_normalize(x: Tensor, dim: int = 1, eps: float = 1e-7):
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def _gn(ch: int):
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)


class UpBlock(nn.Module):
    """ConvTranspose2d 4×4 /2 + GN + ReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.gn = _gn(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor):
        return self.act(self.gn(self.up(x)))


# -----------------------------------------------------------------------------
#  main network
# -----------------------------------------------------------------------------
class PrototypeNet(nn.Module):
    def __init__(self, n_way: int, pretrained: bool = True):
        super().__init__()
        self.n_way = n_way

        # backbone: ResNet18 到 layer2, stride = 8
        base = resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,  # 1/4
            base.layer1,                                    # 1/4
            base.layer2                                     # 1/8  (C=128)
        )
        self.feat_channels = 128

        # decoder: 32→64→128→256, no skip
        self.dec = nn.Sequential(
            UpBlock(n_way, 64),      # 32→64
            UpBlock(64, 32),         # 64→128
            UpBlock(32, n_way),      # 128→256
        )

    # ------------------------------ private utils ---------------------------
    def _extract(self, imgs: Tensor) -> Tensor:
        feats = self.encoder(imgs)                # B×128×32×32
        return l2_normalize(feats, 1)

    # ------------------------------ forward ---------------------------------
    def forward(
        self,
        support_set: List[Tuple[Tensor, Tensor, int]],
        query_set:   List[Tuple[Tensor, Tensor]],
        selected_classes: List[int]
    ) -> Tensor:
        # stack
        s_imgs  = torch.stack([img for img, _, _ in support_set])
        s_masks = torch.stack([mask for _, mask, _ in support_set])
        q_imgs  = torch.stack([img for img, _ in query_set])

        # features
        s_feat = self._extract(s_imgs)   # S×128×32×32
        q_feat = self._extract(q_imgs)   # Q×128×32×32

        S, C, h, w = s_feat.shape
        prototypes = []
        for cls in selected_classes:
            m = (s_masks == cls).float()
            m_ds = F.interpolate(m.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
            area = m_ds.sum((1, 2))
            valid = area > 0
            if valid.any():
                per = (s_feat * m_ds.unsqueeze(1)).sum((2, 3)) / (area.unsqueeze(1) + 1e-7)
                proto = per[valid].mean(0)
            else:
                proto = torch.zeros(C, device=s_feat.device)
            prototypes.append(proto)
        prototypes = l2_normalize(torch.stack(prototypes), 1)  # (n_way,128)

        sims = torch.einsum('bchw,nc->bnhw', q_feat, prototypes)  # Q×n_way×32×32
        logits = self.dec(sims)                                   # Q×n_way×256×256
        return logits


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    net = PrototypeNet(5, pretrained=False)
    sup = [(torch.randn(3,256,256), torch.randint(0,20,(256,256)), 4) for _ in range(5)]
    que = [(torch.randn(3,256,256), torch.randint(0,20,(256,256))) for _ in range(2)]
    o = net(sup, que, [4,6,8,10,12])
    print(o.shape)  # (2,5,256,256)
