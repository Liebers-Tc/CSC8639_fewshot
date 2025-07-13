import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from model.prototype_model import PrototypeNet
from utils.dataloader import EpisodeDataset
from utils.loss import get_loss
from utils.metrics import get_metric
from utils.optim import get_optimizer, get_scheduler
from utils.savepath import get_save_path
from utils.compute import compute_mean_std
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="FewShot Segmentation Training")

    parser.add_argument('--split_class_json', type=str, required=True)
    parser.add_argument('--class2images_mapping_json', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    
    parser.add_argument('--model_name', type=str, default='prototype_resnet18')

    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--q_query', type=int, default=5)
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--val_episodes', type=int, default=10)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, required=True)

    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--metric', nargs='+', default=['miou', 'dice', 'acc'])
    parser.add_argument('--ignore_index', type=int, default=255)  # background pixel value
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--main_metric', type=str, default='loss')
    
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--is_resume',  action='store_true')
    parser.add_argument('--use_amp', action='store_true')

    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    
    return parser.parse_args()


def episodic_collate(batch):
    # 目前只对 batch size = 1 时有效
    # 解决 dataloader 对齐批次时添加 batch dim 的问题
    return batch[0]

def main():
    args = parse_args()

    # init wandb_logger
    if args.wandb:
        from utils.wandb_utils import WandbLogger
        wandb_logger = WandbLogger(project="CSC8639_FewShot", run_name=Path(args.save_dir).name, config=vars(args))
    else:
        wandb_logger = None

    # Auto create save_dir
    save_dir = args.save_dir or get_save_path(root_dir='result')

    # Dataloader
    # mean, std = compute_mean_std(Path(args.img_dir))
    train_loader = DataLoader(
        EpisodeDataset(split_class_json=args.split_class_json, 
                       class2images_mapping_json=args.class2images_mapping_json, 
                       img_dir=args.img_dir, 
                       mask_dir=args.mask_dir, 
                       n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query, episodes=args.train_episodes, 
                       phase="train"), 
        batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=episodic_collate)
    
    val_loader = DataLoader(
        EpisodeDataset(split_class_json=args.split_class_json, 
                       class2images_mapping_json=args.class2images_mapping_json, 
                       img_dir=args.img_dir, 
                       mask_dir=args.mask_dir, 
                       n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query, episodes=args.val_episodes, 
                       phase="val"), 
        batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=episodic_collate)

    # Model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'
    if args.model_name == 'prototype_resnet18':
        model = PrototypeNet(n_way=args.n_way).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Loss & Metrics
    loss_fn = get_loss(name=args.loss, ignore_index=args.ignore_index)
    metric_fn = get_metric(names=args.metric, prefix='', per_image=False, ignore_index=args.ignore_index) # 可选返回每张图的指标

    # Optimizer & Scheduler
    optimizer = get_optimizer(model=model, name=args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay, use_param_groups=False) # 可选参数分组
    main_metric = args.main_metric or args.metric[0]
    if args.scheduler == 'plateau':
        scheduler = get_scheduler(optimizer=optimizer, name=args.scheduler, 
                                # plateau kwargs
                                main_metric=main_metric, patience=6, factor=0.1, threshold=1e-5, min_lr=1e-7)
    else:
        scheduler = get_scheduler(optimizer=optimizer, name=args.scheduler)
    
    # Trainer
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      epochs=args.epochs,
                      loss_fn=loss_fn,
                      metric_fn=metric_fn,
                      ignore_index=args.ignore_index,
                      main_metric=main_metric,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      early_stopping_patience=args.early_stopping_patience,
                      device=device,
                      use_amp=args.use_amp,
                      wandb_logger=wandb_logger,
                      save_dir=save_dir,
                      weight_path=args.weight_path,
                      is_resume = args.is_resume
                      )

    # Start Training
    trainer.load_checkpoint()

    msg = f"[CONFIG] n_way={args.n_way}, k_shot={args.k_shot}, q_query={args.q_query}, train_episodes={args.train_episodes}, val_episodes={args.val_episodes}, ignore_index={args.ignore_index}\n\n"
    print(msg)
    trainer.logger.log_text(msg)

    trainer.fit()

    if args.wandb:
        wandb_logger.finish()

if __name__ == '__main__':
    main()