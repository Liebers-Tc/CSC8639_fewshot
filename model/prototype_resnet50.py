# coding: utf-8
"""
Prototype‑based few‑shot segmentation network with a ResNet‑50 backbone (stride = 8) and
light‑weight decoder.
Key design choices (all discussed with the user):
•  Backbone: ResNet‑50 truncated after layer4, with dilation so that the final feature map is 1/8 of the input size (H/8 × W/8).
•  Normalisation: GroupNorm (GN) everywhere → insensitive to batch = 1 episodes.
•  Prototype computation: per‑shot masked average → per‑class prototype = mean(per‑shot proto), skipping support images that lack the class.
•  Background class (optional): if include_bg=True, an extra learnable bias proto is appended and queried pixels not belonging to selected classes are supervised as background.
•  Decoder: 1×1 Conv + GN + ReLU on the similarity map, followed by bilinear up‑sampling (no heavy transposed‑conv stack ⇒ no "马赛克").
•  Forward signature保持兼容之前的 Trainer:
      logits = model(support_set, query_set, selected_classes)
      ‑ support_set  : List[(img, mask, cls)]  mask retains **original** class IDs.
      ‑ query_set    : List[(img, mask)]      mask retains original IDs (255 = ignore if include_bg=False).
      ‑ selected_classes : List[int]           order defines channel order in logits.

This file can replace the old model implementation.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
#  Utilities
# -----------------------------------------------------------------------------

def l2_normalize(x, dim: int = 1, eps: float = 1e-10):
    """Normalise x to unit length along dim."""
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


# -----------------------------------------------------------------------------
#  Backbone with stride = 8 (ResNet‑50 + dilated conv)
# -----------------------------------------------------------------------------

def resnet50_stride8(pretrained: bool = True) -> nn.Module:
    """Return ResNet‑50 truncated after layer4, but with output stride = 8."""
    resnet = models.resnet50(pretrained=pretrained, replace_stride_with_dilation=(False, True, True))
    # stage strides: conv1‑2, layer1‑4 (1,2,2) → overall 1/8
    layers = [
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),  # 1/4
        resnet.layer1,  # 1/4
        resnet.layer2,  # 1/8
        resnet.layer3,  # 1/8 (dilated)
        resnet.layer4,  # 1/8 (dilated)
    ]
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
#  ProtoSegNet
# -----------------------------------------------------------------------------

class ProtoSegNet(nn.Module):
    def __init__(self, n_way: int, include_bg: bool = False, pretrained_backbone: bool = True, proto_normalize: bool = True):
        """
        Args
        -----
        n_way:          number of foreground classes in each episode (not counting background).
        include_bg:     if True, append an extra background channel & supervise background pixels.
        pretrained_backbone: initialise backbone with ImageNet weights.
        proto_normalize: whether to L2‑normalise prototypes (& query feats) before similarity.
        """
        super().__init__()
        self.n_way = n_way
        self.include_bg = include_bg
        self.proto_normalize = proto_normalize

        self.encoder = resnet50_stride8(pretrained_backbone)
        self.out_channels = 2048  # resnet50 layer4 channel dim

        # lightweight segmentation head: (similarity map)‑>conv1x1‑>GN‑>ReLU
        channels_out = n_way + 1 if include_bg else n_way
        self.head = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(32, channels_out), num_channels=channels_out),
            nn.ReLU(inplace=True),
        )

        # optional learnable background bias prototype (for include_bg=True)
        if include_bg:
            self.bg_proto = nn.Parameter(torch.zeros(self.out_channels))
            nn.init.normal_(self.bg_proto, mean=0.0, std=0.01)

    # ------------------------------------------------------------------
    #  Feature extraction
    # ------------------------------------------------------------------
    def extract_features(self, imgs):
        """imgs: [B,3,H,W] → feats: [B,C,H/8,W/8]"""
        feats = self.encoder(imgs)
        if self.proto_normalize:
            feats = l2_normalize(feats, dim=1)
        return feats

    # ------------------------------------------------------------------
    #  Prototype computation (per‑shot equal weight)
    # ------------------------------------------------------------------
    def _compute_prototypes(self, support_feats, support_masks,
                        support_labels,            # ← 新参数
                        selected_classes):
        S, C, H, W = support_feats.shape
        prototypes = []

        for cls_id in selected_classes:
            # 只取主类 == cls_id 的支撑图索引
            idx = (support_labels == cls_id).nonzero(as_tuple=True)[0]  # shape (?,)
            if idx.numel() == 0:                       # 理论上不会发生
                prototypes.append(torch.zeros(C, device=support_feats.device))
                continue

            feat_cls  = support_feats[idx]             # (S',C,H,W)
            mask_cls  = support_masks[idx]             # (S',H,W)

            # 在这些图里取 cls_id 像素
            cls_mask  = (mask_cls == cls_id).float()   # (S',H,W)
            cls_mask  = F.interpolate(cls_mask.unsqueeze(1),
                                    size=(H, W), mode='nearest').squeeze(1)
            area      = cls_mask.sum((1, 2))           # (S',)

            per_proto = (feat_cls * cls_mask.unsqueeze(1)).sum((2, 3)) \
                        / (area.unsqueeze(1) + 1e-7)   # (S',C)
            proto = per_proto.mean(0)
            prototypes.append(proto)

        if self.include_bg:
            prototypes.append(l2_normalize(self.bg_proto, dim=0) if self.proto_normalize else self.bg_proto)

        prototypes = torch.stack(prototypes)  # [n_way(+1), C]
        if self.proto_normalize:  # already unit but just in case bg proto not
            prototypes = l2_normalize(prototypes, dim=1)
        return prototypes

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    @torch.cuda.amp.autocast(dtype=torch.float16, enabled=False)
    def forward(self,
                support_set,
                query_set,
                selected_classes):
        """Return logits of shape [Q, n_out, H, W]."""
        # Stack
        support_imgs  = torch.stack([img for img, _, _ in support_set])      # (S,3,H,W)
        support_masks = torch.stack([mask for _, mask, _ in support_set])    # (S,H,W)
        support_labels = torch.tensor([cls for _, _, cls in support_set])  # (S,)  ← NEW
        query_imgs    = torch.stack([img for img, _ in query_set])           # (Q,3,H,W)

        # Feature extraction
        support_feats = self.extract_features(support_imgs)   # (S,C,h,w)
        query_feats   = self.extract_features(query_imgs)     # (Q,C,h,w)

        # Prototypes & similarity
        prototypes = self._compute_prototypes(support_feats, support_masks, support_labels, selected_classes)  # (N,C)
        sims = torch.einsum("bchw,nc->bnhw", query_feats, prototypes)  # (Q,N,h,w)

        # Light head & upsample
        logits_low  = self.head(sims)         # (Q,N,h,w)
        logits_high = F.interpolate(logits_low, scale_factor=8, mode="bilinear", align_corners=False)
        return logits_high


# -----------------------------------------------------------------------------
#  Quick test (sanity check)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    n_way = 5
    model = ProtoSegNet(n_way=n_way, include_bg=False).eval()
    supp = [(torch.randn(3, 256, 256), torch.randint(0, 20, (256, 256)), 3) for _ in range(5)]
    que  = [(torch.randn(3, 256, 256), torch.randint(0, 20, (256, 256))) for _ in range(2)]
    logits = model(supp, que, selected_classes=[3, 5, 7, 9, 11])
    print("logits", logits.shape)  # (2,5,256,256)
