# coding: utf-8
"""
Few‑Shot UNet
-------------
• **Encoder/Decoder** directly reuse the standard UNet parts you provided.
• Compute class prototypes at the bottleneck feature (1024 channels, 1/16 res).
• For each query, similarity maps (n_way) → 1×1 conv projection to 1024 ch so
  that the original Up blocks can be used with skip‑connections intact.
• Forward interface matches Trainer: `forward(support_set, query_set,
  selected_classes)` → logits `[Q, n_way, 256, 256]`.
"""
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts_git import DoubleConv, Down, Up, OutConv  # assume same folder

Tensor = torch.Tensor


# ------------------------------------------------ helpers -----------------------------------------

def l2norm(x: Tensor, dim: int = 1, eps: float = 1e-7) -> Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


# ------------------------------------------------ model -------------------------------------------
class FewShotUNet(nn.Module):
    def __init__(self, n_way: int, bilinear: bool = False):
        super().__init__()
        self.n_way = n_way
        self.bilinear = bilinear

        # ------- encoder (same as original UNet) -------
        self.inc   = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        fact = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // fact)     # 1024

        # ------- projection from sim‑map n_way -> 1024 -------
        self.proj = nn.Conv2d(n_way, 1024 // fact, kernel_size=1, bias=False)

        # ------- decoder blocks (reuse Up) -------
        self.up1 = Up(1024, 512 // fact, bilinear)
        self.up2 = Up(512, 256 // fact, bilinear)
        self.up3 = Up(256, 128 // fact, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_way)

    # -------------------------------- private --------------------------------
    def _encode(self, x: Tensor):
        x1 = self.inc(x)       # 64   256
        x2 = self.down1(x1)    # 128  128
        x3 = self.down2(x2)    # 256   64
        x4 = self.down3(x3)    # 512   32
        x5 = self.down4(x4)    # 1024  16
        return (x1, x2, x3, x4, x5)

    # -------------------------------- forward --------------------------------
    def forward(self,
                support_set: List[Tuple[Tensor, Tensor, int]],
                query_set:   List[Tuple[Tensor, Tensor]],
                selected_classes: List[int]) -> Tensor:

        # ----- stack tensors -----
        s_imgs  = torch.stack([img for img, _, _ in support_set])   # S,3,256,256
        s_masks = torch.stack([mask for _, mask, _ in support_set]) # S,256,256
        q_imgs  = torch.stack([img for img, _ in query_set])        # Q,3,256,256

        # ----- encode -----
        _, _, _, _, s_feat = self._encode(s_imgs)  # S,1024,16,16
        q_x1, q_x2, q_x3, q_x4, q_x5 = self._encode(q_imgs)

        # L2‑normalise deep feats
        s_feat = l2norm(s_feat, 1)
        q_feat = l2norm(q_x5, 1)

        S, C, h, w = s_feat.shape
        prototypes = []
        for cls in selected_classes:
            m = (s_masks == cls).float()
            m_ds = F.interpolate(m.unsqueeze(1), (h, w), mode='nearest').squeeze(1)
            area = m_ds.sum((1, 2))
            valid = area > 0
            if valid.any():
                per = (s_feat * m_ds.unsqueeze(1)).sum((2, 3)) / (area.unsqueeze(1) + 1e-7)
                proto = per[valid].mean(0)
            else:
                proto = torch.zeros(C, device=s_feat.device)
            prototypes.append(proto)
        prototypes = l2norm(torch.stack(prototypes), 1)  # (n_way,1024)

        # ----- similarity -----
        sims = torch.einsum('bchw,nc->bnhw', q_feat, prototypes)  # Q,n_way,16,16

        # ----- project & decode -----
        z = self.proj(sims)    # Q,1024,16,16
        z = self.up1(z, q_x4)  # 32
        z = self.up2(z, q_x3)  # 64
        z = self.up3(z, q_x2)  # 128
        z = self.up4(z, q_x1)  # 256
        logits = self.outc(z)
        return logits


# quick smoke‑test
if __name__ == "__main__":
    net = FewShotUNet(n_way=5)
    sup = [(torch.randn(3,256,256), torch.randint(0,20,(256,256)), 4) for _ in range(5)]
    que = [(torch.randn(3,256,256), torch.randint(0,20,(256,256))) for _ in range(2)]
    out = net(sup, que, [4,6,8,10,12])
    print(out.shape)  # (2,5,256,256)
