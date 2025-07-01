
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class PrototypeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=104, backbone='resnet18', pretrained=True):
        super().__init__()
        self.num_classes = num_classes

        # Use ResNet18 as encoder
        resnet = resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.out_channels = 256  # resnet layer3 output

    def extract_features(self, images):
        """Input: Tensor [B, 3, H, W] -> Output: [B, C, h, w]"""
        return self.encoder(images)

    def compute_prototype(self, support_feats, support_masks):
        """
        support_feats: [N*K, C, h, w]
        support_masks: [N*K, h, w] with 0/1 mask
        Return: prototype [N, C]
        """
        n_shots = len(support_feats)
        c = support_feats.shape[1]
        h, w = support_feats.shape[2:]

        # Flatten support features
        feats = support_feats.view(n_shots, c, -1)                # [NK, C, h*w]
        masks = support_masks.view(n_shots, -1).float()           # [NK, h*w]

        # Apply mask to feature map
        masked_feats = feats * masks.unsqueeze(1)                # [NK, C, h*w]
        prototype = masked_feats.sum(dim=2) / (masks.sum(dim=1, keepdim=True) + 1e-5)  # [NK, C]
        return prototype.mean(dim=0, keepdim=True)                # [1, C]

    def forward(self, support_set, query_set):
        """
        support_set: list of (image, mask) tensors
        query_set: list of (image, mask) tensors
        Returns:
            query_pred: Tensor [Q, num_classes, h, w]
        """
        support_imgs = torch.stack([img for img, _ in support_set], dim=0)   # [N*K, 3, H, W]
        support_masks = torch.stack([mask for _, mask in support_set], dim=0)  # [N*K, H, W]
        query_imgs = torch.stack([img for img, _ in query_set], dim=0)       # [Q, 3, H, W]

        support_feats = self.extract_features(support_imgs)  # [NK, C, h, w]
        query_feats = self.extract_features(query_imgs)      # [Q, C, h, w]

        # compute prototype
        proto = self.compute_prototype(support_feats, support_masks)   # [1, C]
        proto = F.normalize(proto, dim=1)                              # cosine normalize

        # query
        q = query_feats.shape[0]
        query_feats = F.normalize(query_feats, dim=1)
        query_feats_flat = query_feats.view(q, self.out_channels, -1)  # [Q, C, h*w]

        # cosine similarity
        sim = torch.einsum('qcm,pc->qpm', query_feats_flat, proto)     # [Q, 1, h*w]
        sim = sim.view(q, 1, query_feats.shape[2], query_feats.shape[3])  # [Q, 1, h, w]

        # expand to multi-class prediction if needed
        if self.num_classes > 1:
            out = torch.zeros((q, self.num_classes, sim.shape[2], sim.shape[3]), device=sim.device)
            out[:, 1] = sim[:, 0]  # class 1 foreground
            out[:, 0] = -sim[:, 0]  # class 0 background
            return out
        else:
            return sim  # [Q, 1, h, w]
