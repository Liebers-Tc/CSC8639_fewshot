import os
import matplotlib.pyplot as plt
from IPython.display import display

class Logger:
    def __init__(self, save_dir, save=True, show=False, wandb_logger=None):
        """
        save_dir (str): 图像和日志文件的保存目录
        save (bool): 是否将图像保存到本地
        show (bool): 是否弹窗显示图像（调试用）
        wandb_logger (WandbLogger or None): 可选的 wandb 日志对象，控制是否上传到 wandb
        """

        self.save_dir = save_dir
        self.save = save
        self.show = show
        self.wandb_logger = wandb_logger
        os.makedirs(save_dir, exist_ok=True)

    def log_text(self, line):
        """记录文本日志到本地 log.txt"""
        log_path = os.path.join(self.save_dir, 'log.txt')
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    def log_scalar(self, tag, value, step):
        """记录标量（如 loss、acc）到 wandb（如启用）"""
        if self.wandb_logger:
            self.wandb_logger.log({tag: value}, step=step)

    def plot_curve(self, x, y1, y2=None, title='', ylabel='', labels=None):
        """仅绘图，返回 matplotlib Figure 对象（不保存、不上传）"""
        fig, ax = plt.subplots()
        ax.plot(x, y1, label=labels[0] if labels else 'Line 1')
        if y2:
            ax.plot(x, y2, label=labels[1] if labels and len(labels) > 1 else 'Line 2')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        return fig

    def save_curve(self, train_losses, val_losses, val_metrics_dict=None):
        """保存训练/验证损失和指标曲线，可选上传 wandb"""
        epochs = range(1, len(train_losses) + 1)

        # 损失曲线
        fig = self.plot_curve(
            x=epochs,
            y1=train_losses,
            y2=val_losses,
            title='Loss Curve',
            ylabel='Loss',
            labels=['Train Loss', 'Val Loss']
            )
        
        loss_path = os.path.join(self.save_dir, 'loss_curve.png')
        if self.save:
            fig.savefig(loss_path)
        if self.wandb_logger:
            self.wandb_logger.log({'Loss Curve': self.wandb_logger.image(loss_path)})
        if self.show:
            display(fig)
        plt.close(fig)

        # 验证指标曲线
        if val_metrics_dict:
            for metric_name, val_values in val_metrics_dict.items():
                fig = self.plot_curve(
                    x=epochs,
                    y1=val_values,
                    title=f"Val {metric_name} Curve",
                    ylabel=metric_name.upper(),
                    labels=[f"Val {metric_name}"]
                    )

                metric_path = os.path.join(self.save_dir, f"val_{metric_name.lower()}_curve.png")
                if self.save:
                    fig.savefig(metric_path)
                if self.wandb_logger:
                    self.wandb_logger.log({f"Val {metric_name} Curve": self.wandb_logger.image(metric_path)})
                if self.show:
                    display(fig)
                plt.close(fig)