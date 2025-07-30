import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from model.resnet50_8x_dot import ProtoNet as resnet50_8x_dot
from model.resnet50_8x_euclid import ProtoNet as resnet50_8x_euclid
from model.resnet50_8x_cosine import ProtoNet as resnet50_8x_cosine

from model.resnet50_8x_cosine_simp_bgmix import ProtoNet as resnet50_8x_cosine_simp_bgmix
from model.resnet50_8x_cosine_accu_bgmix import ProtoNet as resnet50_8x_cosine_accu_bgmix

from model.resnet50_8x_cosine_simp_bgmix_scaler import ProtoNet as resnet50_8x_cosine_simp_bgmix_scaler
from model.resnet50_8x_cosine_accu_bgmix_scaler import ProtoNet as resnet50_8x_cosine_accu_bgmix_scaler

from utils.dataloader import EpisodeDataset
from utils.loss import get_loss
from utils.metrics import get_metric
from utils.optim import get_optimizer, get_scheduler
from utils.savepath import get_save_path
from utils.compute import compute_mean_std
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="FewShot Segmentation Training")

    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--split_class_json', type=str, required=True)
    parser.add_argument('--class2images_mapping_json', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--beta_mix', type=float, default=0.25)
    parser.add_argument('--alpha_bg', type=float, default=0.75)

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
    # parser.add_argument('--bg_weight', type=float, default=0.25)  # loss中背景类别权重
    parser.add_argument('--metric', nargs='+', default=['miou', 'dice', 'acc'])
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--main_metric', type=str, default='loss')
    
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--is_resume',  action='store_true')
    parser.add_argument('--use_amp', action='store_true')

    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    
    return parser.parse_args()


def episodic_collate(batch):
    # 只对 batch size = 1 时有效
    # 解决 dataloader 对齐批次时添加 batch dim 的问题
    return batch[0]

def main():
    args = parse_args()

    # init seed
    if args.seed is not None:
        from utils.seed import set_global_seed, set_worker_seed
        set_global_seed(args.seed)
        worker_seed = set_worker_seed(args.seed)
    else:
        worker_seed = None

    # init wandb_logger
    if args.wandb:
        from utils.wandb_utils import WandbLogger
        wandb_logger = WandbLogger(project="CSC8639_FewShot_FoodSeg103", run_name=Path(args.save_dir).name, config=vars(args))
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
        batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=episodic_collate,
        pin_memory=True, worker_init_fn=worker_seed, persistent_workers=True)
    
    val_loader = DataLoader(
        EpisodeDataset(split_class_json=args.split_class_json, 
                       class2images_mapping_json=args.class2images_mapping_json, 
                       img_dir=args.img_dir, 
                       mask_dir=args.mask_dir,
                       n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query, episodes=args.val_episodes, 
                       phase="val"), 
        batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=episodic_collate,
        pin_memory=True, worker_init_fn=worker_seed, persistent_workers=False)

    # Model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'
    if args.model_name == 'resnet50_8x_dot':
        model = resnet50_8x_dot(n_way=args.n_way).to(device)
    elif args.model_name == 'resnet50_8x_euclid':
        model = resnet50_8x_euclid(n_way=args.n_way).to(device)
    elif args.model_name == 'resnet50_8x_cosine':
        model = resnet50_8x_cosine(n_way=args.n_way).to(device)
    elif args.model_name == 'resnet50_8x_cosine_simp_bgmix':
        model = resnet50_8x_cosine_simp_bgmix(n_way=args.n_way, beta_mix=args.beta_mix).to(device)
    elif args.model_name == 'resnet50_8x_cosine_accu_bgmix':
        model = resnet50_8x_cosine_accu_bgmix(n_way=args.n_way, beta_mix=args.beta_mix).to(device)
    elif args.model_name == 'resnet50_8x_cosine_simp_bgmix_scaler':
        model = resnet50_8x_cosine_simp_bgmix_scaler(n_way=args.n_way, beta_mix=args.beta_mix, alpha_bg=args.alpha_bg).to(device)
    elif args.model_name == 'resnet50_8x_cosine_accu_bgmix_scaler':
        model = resnet50_8x_cosine_accu_bgmix_scaler(n_way=args.n_way, beta_mix=args.beta_mix, alpha_bg=args.alpha_bg).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Loss & Metrics
    # class_weight = torch.ones(args.n_way + 1)  # 前景权重 1
    # class_weight[-1] = args.bg_weight  # 让背景权重 < 前景权重
    # class_weight = class_weight.to(device)
    class_weight = None
    loss_fn = get_loss(name=args.loss, weight=class_weight)
    metric_fn = get_metric(names=args.metric, prefix='', per_image=False) # 可选返回每张图的指标

    # Optimizer & Scheduler
    optimizer = get_optimizer(model=model, name=args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay, use_param_groups=False) # 可选参数分组
    main_metric = args.main_metric or args.metric[0]
    if args.scheduler == 'cosine':
        scheduler = get_scheduler(optimizer=optimizer, name=args.scheduler, 
                                # cosine kwargs
                                T_max=args.epochs, eta_min=1e-5)
    elif args.scheduler == 'plateau':
        scheduler = get_scheduler(optimizer=optimizer, name=args.scheduler, 
                                # plateau kwargs
                                main_metric=main_metric, patience=5, factor=0.1, threshold=1e-5, min_lr=1e-7)
    else:
        scheduler = get_scheduler(optimizer=optimizer, name=args.scheduler)
    
    # Trainer
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      epochs=args.epochs,
                      loss_fn=loss_fn,
                      metric_fn=metric_fn,
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

    msg = f"[CONFIG] n_way={args.n_way}, k_shot={args.k_shot}, q_query={args.q_query}, train_episodes={args.train_episodes}, val_episodes={args.val_episodes}\n\n"
    print(msg)
    trainer.logger.log_text(msg)

    trainer.fit()

    if args.wandb:
        wandb_logger.finish()

if __name__ == '__main__':
    main()