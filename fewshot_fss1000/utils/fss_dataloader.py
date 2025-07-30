import random
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
import torch


""" ===构建 EpisodeDataset=== """
class EpisodeDataset(Dataset):
    
    # 将 PIL 格式的 mask 图像转换为 Long 类型的 Tensor，并移除可能多余的通道维度
    @staticmethod
    def _mask_to_tensor(m):
        return torch.as_tensor(np.array(m), dtype=torch.long).squeeze()
    
    def __init__(self, dataset_dir,
                 n_way=5, k_shot=5, q_query=5, episodes=1000, phase="train", 
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        
        self.dataset_dir = Path(dataset_dir)
        classes = [f for f in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, f))]  # 只取子文件夹
        classes = sorted(classes)

        # 按照 8:1:1 划分
        total = len(classes)
        n_train = int(0.8 * total)
        n_val = int(0.1 * total)
        if phase == "train":
            self.classes = classes[:n_train]
        elif phase == "val":
            self.classes = classes[n_train:n_train + n_val]
        elif phase == "test":
            self.classes = classes[n_train + n_val:]

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.phase = phase
        self.episodes = episodes
        self.cls2idx = {cls: i for i, cls in enumerate(classes)}

        # 图像预处理：转为 Tensor 并标准化
        self.image_transform = T.Compose([
            T.Resize((256, 256), interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean, std)
            ])
        
        # 掩码预处理：读取为 Long 类型张量（不做归一化或缩放）
        self.mask_transform = T.Compose([
            T.Resize((256, 256), interpolation=Image.NEAREST),
            T.Lambda(self._mask_to_tensor)
            ])

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        # 随机取 n_way 个类别
        selected_classes = random.sample(self.classes, self.n_way)
        
        support_set, query_set = [], []
        for cls in selected_classes:
            path = Path(self.dataset_dir) / cls
            names = [f.stem for f in Path(path).glob("*.jpg")]
            assert len(names) >= self.k_shot + self.q_query, \
                f"{cls} 样本不足"
            sampled = random.sample(names, self.k_shot + self.q_query)
            for name in  sampled[:self.k_shot]:
                img_path = path / f"{name}.jpg"
                mask_path = path / f"{name}.png"

                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                support_img = self.image_transform(img)
                support_mask = self.mask_transform(mask)

                support_set.append((support_img, support_mask, self.cls2idx[cls]))

            for name in sampled[self.k_shot:]:
                img_path = path / f"{name}.jpg"
                mask_path = path / f"{name}.png"

                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                query_img = self.image_transform(img)
                query_mask = self.mask_transform(mask)
                
                query_set.append((query_img, query_mask, self.cls2idx[cls]))

        selected_cls_idx = [self.cls2idx[cls] for cls in selected_classes]
        return support_set, query_set, selected_cls_idx