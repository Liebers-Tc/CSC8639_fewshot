import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display


class Visualizer:
    def __init__(self, save_dir, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                 max_save_samples=None, max_show_groups=3, wandb_logger=None, 
                 upload_pred=False, upload_overlay=False, upload_group=True):
        """
        save_dir (str): 图像保存目录
        mean (list): 图像归一化的均值（用于反归一化）
        std (list): 图像归一化的标准差（用于反归一化）
        max_save_samples (int): 最多保存多少张图像
        max_show_groups (int): 最多展示多少组图像
        wandb_logge (WandbLogger or None): 由主控文件传入 WandbLogger 实例，默认为 None 不开启上传
        upload_pred/overlay/group (bool): 是否上传预测图 / 叠加图 / 组图到 wandb
        """

        self.save_dir = save_dir
        self.mean = mean
        self.std = std
        self.max_save_samples = max_save_samples
        self.max_show_groups = max_show_groups
        self.wandb_logger = wandb_logger
        self.upload_pred = upload_pred
        self.upload_overlay = upload_overlay
        self.upload_group = upload_group
        os.makedirs(save_dir, exist_ok=True)

    def denormalize(self, image):
        """对归一化后的图像进行反标准化，恢复到原始图像色彩"""
        mean = torch.tensor(self.mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(self.std).view(3, 1, 1).to(image.device)
        return image * std + mean

    def pred_image(self, mask):
        """将 mask 转换为彩色图像用于可视化"""
        return mask.cpu().numpy()  # RGB mask

    def overlay_image(self, image, mask, alpha=0.5):
        """将预测 mask 叠加到原图上，生成混合可视化图像"""
        image = self.denormalize(image).cpu()
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = self.pred_image(mask)

        mask_bin = (mask_np == 255).astype(np.float32)[..., None]  # H x W x 1
        output = image_np * mask_bin  # 前景保留，背景为 0（黑）
        return output.clip(0, 1)

    def group_image(self, image, gt_mask, pred_mask):
        """拼接图像 / 叠加图 / GT mask / Pred mask 生成组图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 6))
        axes = axes.flatten()
        
        image_denorm = self.denormalize(image)
        overlay = self.overlay_image(image, pred_mask)
        pred_vis = self.pred_image(pred_mask)

        axes[0].imshow(image_denorm.permute(1, 2, 0).cpu())
        axes[0].set_title("Image")
        axes[1].imshow(overlay)
        axes[1].set_title("Overlay")
        axes[2].imshow(gt_mask.cpu())
        axes[2].set_title("Ground Truth")
        axes[3].imshow(pred_vis)
        axes[3].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        return fig

    def save_pred(self, preds, start_index=0):
        """保存/上传 预测 mask 可视化图"""
        total = len(preds) if self.max_save_samples is None else min(len(preds), self.max_save_samples)
        for i in range(total):
            idx = start_index + i + 1
            pred_vis = self.pred_image(preds[i])
            save_path = os.path.join(self.save_dir, f"pred_{idx}.png")
            plt.imsave(save_path, pred_vis)
            if self.wandb_logger and self.upload_pred:
                self.wandb_logger.log({f"Pred_Img/Prediction_Img/{idx:03d}_{os.path.basename(save_path)}": self.wandb_logger.image(save_path)})

    def save_overlay(self, images, preds, start_index=0):
        """保存/上传 原图与预测 mask 的叠加图"""
        total = len(images) if self.max_save_samples is None else min(len(images), self.max_save_samples)
        for i in range(total):
            idx = start_index + i + 1
            overlay = self.overlay_image(images[i], preds[i])
            save_path = os.path.join(self.save_dir, f"overlay_{idx}.png")
            plt.imsave(save_path, overlay)
            if self.wandb_logger and self.upload_overlay:
                self.wandb_logger.log({f"Pred_Img/Overlay_Img/{idx:03d}_{os.path.basename(save_path)}": self.wandb_logger.image(save_path)})

    def save_group(self, images, gts, preds, start_index=0):
        """保存/上传 组图（Image / Overlay / GT / Pred）"""
        total = len(images) if self.max_save_samples is None else min(len(images), self.max_save_samples)
        for i in range(total):
            idx = start_index + i + 1
            fig = self.group_image(images[i], gts[i], preds[i])
            save_path = os.path.join(self.save_dir, f"group_{idx}.png")
            fig.savefig(save_path)
            if self.wandb_logger and self.upload_group:
                self.wandb_logger.log({f"Pred_Img/Group_Img/{idx:03d}_{os.path.basename(save_path)}": self.wandb_logger.image(save_path)})
            plt.close(fig)

    def plot_demo(self, images, gts, preds):
        """仅展示指定数量的组图（不保存）"""
        groups = min(self.max_show_groups, len(images))
        for i in range(groups):
            fig = self.group_image(images[i], gts[i], preds[i])
            display(fig)
            plt.close(fig)

    def visualize(self, images, gts, preds, start_index=0):
        """完整可视化流程：保存预测图 / 叠加图 / 拼图，同时打印 demo"""
        self.save_group(images, gts, preds, start_index)
        self.plot_demo(images, gts, preds)
