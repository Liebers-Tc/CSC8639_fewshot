import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class PrototypeNet(nn.Module):
    def __init__(self, in_channels=3, backbone="resnet18", pretrained=True):  # 预留 in_channels 和 backbone，虽然没想好要干吗用
        super().__init__()

        # 加载预训练模型作为 encoder
        resnet = resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.out_channels = 256  # encoder输出通道

    def extract_features(self, images):
        return self.encoder(images)  # [B, 3, H, W] -> [B, C, h, w]
    
    def forward(self, support_set, query_set, selected_classes):
        """
        support_set: List[image, mask]，mask 为二值化后类别ID
        query_set: List[image, mask]，mask 为原始类别ID
        selected_classes: List[int]，当前 episode 中 support set 的类别ID
        """
        device = self.encoder[0].weight.device
        n_way = len(selected_classes)

        # initial & extract features
        support_imgs = torch.stack([img for img, _, _ in support_set], dim=0).to(device)  # [S, 3, H, W]
        support_feats = self.extract_features(support_imgs)  # [S, C, h, w]

        # build prototype
        proto_dict = {cls_id: [] for cls_id in selected_classes}
        for i, (img, binary_mask, cls_id) in enumerate(support_set):
            feat = support_feats[i]  # [C, h, w]
            mask = binary_mask.float().unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
            mask = F.interpolate(mask, size=feat.shape[1:], mode='nearest').squeeze()  # [h, w]

            feat_flat = feat.view(self.out_channels, -1)  # [C, h*w]
            mask_flat = mask.view(-1)  # [h*w]
            if mask_flat.sum() == 0:
                continue

            selected_feat = feat_flat[:, mask_flat > 0]  # [C, N]
            proto = selected_feat.mean(dim=1)            # [C]
            proto_dict[cls_id].append(proto)

        # initial / update prototype for each class (mean & normalize)
        prototypes = []
        for cls_id in selected_classes:
            vecs = proto_dict[cls_id]
            if len(vecs) == 0:
                prototypes.append(torch.zeros(1, self.out_channels, device=device))
            else:
                proto = torch.stack(vecs).mean(dim=0, keepdim=True)
                proto = F.normalize(proto, dim=1)
                prototypes.append(proto)

        prototypes = torch.cat(prototypes, dim=0)  # [n_way, C]

        # query inference
        query_imgs = torch.stack([img for img, _ in query_set], dim=0).to(device)  # [B, 3, H, W]
        query_feats = self.extract_features(query_imgs)  # [B, C, h, w]
        query_feats = F.normalize(query_feats, dim=1)    # 归一化后用于余弦相似度

        # similarity (cos)
        sims = torch.einsum('bchw,nc->bnhw', query_feats, prototypes)  # [B, N_way, h, w]

        return sims  # logits, 每个像素在每个support类上的相似度
