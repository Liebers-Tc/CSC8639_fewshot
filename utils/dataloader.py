import random
import json
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
    
    def __init__(self, split_class_json, class2images_mapping_json, 
                 img_dir, mask_dir, 
                 n_way=5, k_shot=5, q_query=5, phase="train", max_episodes=None, 
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.phase = phase
        self.max_episodes = max_episodes or (100000 if phase=="train" else 1000)

        with open(split_class_json, 'r') as f:
            self.split = json.load(f)
        with open(class2images_mapping_json, 'r') as f:
            self.class2imgs = json.load(f)

        # 获取当前阶段对应的类别
        self.classes = self.split[f"{phase}_classes"]

        # 图像预处理：转为 Tensor 并标准化
        self.image_transform = T.Compose([
            # T.Resize((256, 256), interpolation=Image.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean, std)
            ])
        
        # 掩码预处理：读取为 Long 类型张量（不做归一化或缩放）
        self.mask_transform = T.Compose([
            # T.Resize((256, 256), interpolation=Image.NEAREST),
            T.Lambda(self._mask_to_tensor)
            ])

    def __len__(self):
        return self.max_episodes

    def __getitem__(self, idx):
        # 随机取 n_way 个类别
        selected_classes = random.sample(self.classes, self.n_way)
        
        support_set = []
        query_set = []
        # used_query_imgs = set() 预留后续不同类别间的query去重
        for cls in selected_classes:
            sampled_imgs = random.sample(self.class2imgs[str(cls)], self.k_shot + self.q_query)
            
            for img_id in sampled_imgs[:self.k_shot]:
                img_path = self.img_dir / f"{img_id}.jpg"
                mask_path = self.mask_dir / f"{img_id}.png"

                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path)
                binary_mask = (np.array(mask) == int(cls)).astype(np.uint8) # 转化成二值mask
                mask = Image.fromarray(binary_mask)

                support_img = self.image_transform(img)
                support_mask = self.mask_transform(mask)
                
                support_set.append((support_img, support_mask))

            for img_id in sampled_imgs[:self.q_query]:
                img_path = self.img_dir / f"{img_id}.jpg"
                mask_path = self.mask_dir / f"{img_id}.png"

                image = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path)

                query_img = self.image_transform(image)
                query_mask = self.mask_transform(mask)
                
                query_set.append((query_img, query_mask))

        return support_set, query_set