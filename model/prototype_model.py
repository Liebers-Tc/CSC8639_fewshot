import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class PrototypeNet(nn.Module):
    def __init__(self, n_way, backbone="resnet18", pretrained=True):  # 预留 backbone 做接口
        super().__init__()

        # 加载预训练模型作为 encoder
        resnet = resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, 
            resnet.layer2, 
            resnet.layer3
            )
        self.out_channels = 256  # encoder输出通道

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_way, 64, 2, stride=2),   # 16→32
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, stride=2),      # 32→64
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 2, stride=2),      # 64→128
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, n_way, 2, stride=2),   # 128→256
            )
        
    # L2‑normalize feature maps along channel dim for cosine sim
    @staticmethod
    def _normalize(feats):
        return F.normalize(feats, p=2, dim=1, eps=1e-7)

    def extract_features(self, images):
        feats = self.encoder(images)  # [B, 3, H, W] -> [B, C, h, w]
        feats = self._normalize(feats)
        return feats
    
    def forward(self, support_set, query_set, selected_classes):
        """
        support_set: List[image, mask]，mask 为二值化后类别ID
        query_set: List[image, mask]，mask 为原始类别ID
        selected_classes: List[int]，当前 episode 中 support set 的类别ID
        """
        device = self.encoder[0].weight.device

        # stack support/query
        support_imgs = torch.stack([img for img, _, _ in support_set])  # (S,3,256,256)
        support_masks = torch.stack([mask for _, mask, _ in support_set])  # (S,256,256)
        query_imgs   = torch.stack([img for img, _ in query_set])          # (Q,3,256,256)

        # extract features
        support_feats = self.extract_features(support_imgs)  # (S,256,16,16)
        query_feats   = self.extract_features(query_imgs)    # (Q,256,16,16)

        # build prototype
        prototypes = []
        for cls in selected_classes:
            # 下采样至 (S,16,16)
            cls_mask = (support_masks == cls).float().unsqueeze(1)
            cls_mask = F.interpolate(cls_mask, size=support_feats.shape[-2:], mode='nearest').squeeze(1)

            # ➜ 每张图该类的像素个数   (S,)
            area_per_img = cls_mask.sum((1, 2))

            # ➜ 有效图片布尔索引
            valid_mask   = area_per_img > 0               # (S,)

            if valid_mask.any():                          # 至少 1 张包含该类
                # (1) 先对每张图求 masked mean          (S,C)
                safe_area = area_per_img.clone()
                safe_area[safe_area == 0] = 1             # 防止除 0，但不会被选进平均
                per_img_proto = (support_feats * cls_mask.unsqueeze(1))\
                                .sum((2,3)) / safe_area.unsqueeze(1)

                # (2) 对有效图片做简单平均  (C)
                proto = per_img_proto[valid_mask].mean(0)
            else:
                # 若所有图都缺失该类，用 0 向量占位
                proto = torch.zeros(support_feats.size(1), device=support_feats.device)

            prototypes.append(proto)

        prototypes = torch.stack(prototypes)              # (n_way, C)
        prototypes = self._normalize(prototypes)

        # prototypes = []
        # for cls in selected_classes:
        #     cls_mask = (support_masks == cls).float().unsqueeze(1)
        #     cls_mask = F.interpolate(cls_mask, size=support_feats.shape[-2:], mode='nearest')
        #     cls_mask = cls_mask.squeeze(1)                         # (S,16,16)

        #     # 先对每张图求 masked mean
        #     masked_feats = support_feats * cls_mask.unsqueeze(1)   # (S,C,16,16)
        #     per_img_proto = masked_feats.sum((2,3)) / (cls_mask.sum((1,2)) + 1e-5)  # (S,C)

        #     proto = per_img_proto.mean(0)                          # 再对 S 取平均
        #     prototypes.append(proto)

        # prototypes = torch.stack(prototypes)
        # prototypes = self._normalize(prototypes)

        # similarity (cos)
        sims = torch.einsum('bchw,nc->bnhw', query_feats, prototypes)  # [B, N_way, h, w]
        logits = self.decoder(sims)

        return logits  # logits, 按 sim 解码成原尺寸图像
