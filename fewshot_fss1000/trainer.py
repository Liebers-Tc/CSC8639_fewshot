import os
import torch
from torch.amp import autocast, GradScaler
from utils.log import Logger
from utils.mask_mapping import remap_querymask


class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs, 
                 loss_fn, metric_fn, main_metric, 
                 optimizer, scheduler, 
                 early_stopping_patience, 
                 device, use_amp, 
                 wandb_logger,
                 save_dir, weight_path, is_resume
                 ):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.main_metric = main_metric
        self.greater_is_better = main_metric != 'loss'

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.early_stop_counter = 0

        self.device = device
        self.use_amp = use_amp
        self.wandb_logger = wandb_logger
        self.weight_path = weight_path
        self.is_resume = is_resume

        self.scaler = GradScaler(enabled=use_amp)
        self.best_score = -float('inf') if self.greater_is_better else float('inf')

        self.ckpt_dir = os.path.join(save_dir, 'checkpoint')
        self.log_dir = os.path.join(save_dir, 'log')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = Logger(save_dir=self.log_dir, save=True, show=False, wandb_logger=self.wandb_logger)

        self.train_history = {"loss": []}
        self.val_history = {"loss": []}
        # self.train_metrics = {}
        self.val_metrics = {}


    def run_one_epoch(self, train=True):
        dataloader = self.train_loader if train else self.val_loader
        self.model.train() if train else self.model.eval()

        total_loss = 0.0
        total_metric = {}

        with torch.set_grad_enabled(train):
            for i, (support_set, query_set, selected_classes) in enumerate(dataloader):
                self.optimizer.zero_grad()
                
                support_set = [(img.to(self.device), mask.to(self.device), cls) for img, mask, cls in support_set]
                query_set = [(img.to(self.device), mask.to(self.device), cls) for img, mask, cls in query_set]
                query_masks = torch.stack([mask for _, mask, _ in query_set])
                query_labels = [cls for _, _, cls in query_set]

                with autocast(device_type=self.device, enabled=self.use_amp):
                    outputs = self.model(support_set, query_set, selected_classes)  # model.forward(support_set, query_set, selected_classes)
                    
                    # 将 query mask 中原始类别ID映射为对应索引值，然后进行 loss 和 metrics 计算
                    remapped_querymask = remap_querymask(query_masks, query_labels, selected_classes)
                    # loss 接收 outputs(logits)
                    loss = self.loss_fn(outputs, remapped_querymask)

                # metrics 接收 outputs(logits)，但内部用 argmax 转换成离散值
                # autocast 外部执行 metrics 计算
                preds = torch.argmax(outputs, dim=1)
                metrics = self.metric_fn(preds, remapped_querymask, list(range(len(selected_classes)))) if not train else {}  # 只在 val 阶段生效且不计算背景类
                    
                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()

                for k, v in metrics.items():
                    total_metric[k] = total_metric.get(k, 0.0) + v.item()

        avg_loss = total_loss / len(dataloader)
        avg_metrics = {k: v / len(dataloader) for k, v in total_metric.items()}
        return {"loss": avg_loss, **avg_metrics}
    
    def save_checkpoint(self, epoch, filename):
        filepath = os.path.join(self.ckpt_dir, f"model_{filename}.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            # "train_history": self.train_history,
            # "val_history": self.val_history,
            # "best_score": self.best_score
            }, filepath)
        print(f"Model saved to {filepath}")
        self.logger.log_text(f"Model saved to {filepath}\n")

    def load_checkpoint(self):

        # 0. 权重路径不存在
        if self.weight_path and not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"Checkpoint or Pre-trained weight not found at: {self.weight_path}")
        
        # 1. 断点续训，恢复 epoch 与优化器
        elif self.weight_path and self.is_resume:
            checkpoint = torch.load(self.weight_path, map_location=self.device)
            self.start_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            # self.train_history = checkpoint.get("train_history", {"loss": []})
            # self.val_history = checkpoint.get("val_history", {"loss": []})
            # self.best_score = checkpoint.get("best_score", self.best_score)
            print(f"Resume Model and Hyperparameters from: {self.weight_path}")
            self.logger.log_text(f"Resume Model and Hyperparameters from: {self.weight_path}\n")

        # 2. 仅加载权重，不恢复优化器
        elif self.weight_path and not self.is_resume:
            checkpoint = torch.load(self.weight_path, map_location=self.device)
            self.start_epoch = 0
            self.model.load_state_dict(checkpoint["model_state_dict"])

            print(f"Only resume Model from: {self.weight_path}")
            self.logger.log_text(f"Only resume Model from: {self.weight_path}\n")

        # 3. # 训练新模型
        elif self.weight_path is None:
            self.start_epoch = 0
            print(f"Start training New Model")
            self.logger.log_text(f"Start training New Model\n")
            return

    def fit(self):
        if self.start_epoch >= self.epochs:
            raise ValueError(
                f"Resume epoch ({self.start_epoch}) >= target epochs ({self.epochs})\n"
                f"Increase --epochs above {self.start_epoch} to continue training."
            )
        
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            print(f"\n[Epoch {epoch}/{self.epochs}]")

            train_stats = self.run_one_epoch(train=True)
            val_stats = self.run_one_epoch(train=False)

            train_loss = train_stats["loss"]
            val_loss = val_stats["loss"]

            self.train_history["loss"].append(train_loss)
            self.val_history["loss"].append(val_loss)

            self.logger.log_scalar("Loss/train", train_loss, step=epoch)
            self.logger.log_scalar("Loss/val", val_loss, step=epoch)

            for k, v in val_stats.items():
                if k != "loss":
                    self.val_metrics.setdefault(k, []).append(v) # append val metrics to self.val_metrics
                    self.logger.log_scalar(f"{k}/val", v, step=epoch)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_stats.get(self.main_metric, val_stats['loss']))
                else:
                    self.scheduler.step()

            log_msg = f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, " + \
                      ", ".join([f"{k}={v:.4f}" for k, v in val_stats.items() if k != 'loss']) + \
                      f"\nCurrent learning rate: {self.scheduler.get_last_lr()}"
            print(log_msg)
            self.logger.log_text(log_msg)
            
            main_value = val_stats[self.main_metric]
            value = (main_value > self.best_score) if self.greater_is_better else (main_value < self.best_score)
            if value:
                self.best_score = main_value
                self.early_stop_counter = 0
                self.save_checkpoint(epoch, filename="best")
                best_msg = "Best model updated"
                print(best_msg)
                self.logger.log_text(best_msg)
            else:
                self.early_stop_counter += 1
                if self.early_stopping_patience and self.early_stop_counter >= self.early_stopping_patience:
                    early_msg = "Early stopping triggered"
                    print(early_msg)
                    self.logger.log_text(early_msg)
                    break

        self.save_checkpoint(epoch, filename="final")
        self.logger.save_curve(train_losses=self.train_history["loss"], val_losses=self.val_history["loss"], val_metrics_dict=self.val_metrics)




