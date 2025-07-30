import os
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

from utils.fss_dataloader import EpisodeDataset
from utils.visualization import Visualizer
from utils.metrics import get_metric
from utils.savepath import find_latest_path
from utils.compute import compute_mean_std
from utils.mask_mapping import remap_querymask, reverse_remap


def parse_args():
    parser = argparse.ArgumentParser(description="FewShot Segmentation Prediction")

    parser.add_argument('--seed', type=int, default=None)

    parser.add_argument('--dataset_dir', type=str, default="../data/fewshot_data")

    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--alpha_bg', type=float, default=0.75)
    parser.add_argument('--beta_mix', type=float, default=0.5)

    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--q_query', type=int, default=5)
    parser.add_argument('--pred_episodes', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--metric', nargs='+', default=['miou', 'dice', 'acc'])

    parser.add_argument('--weight_path', type=str, required=True)
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
        wandb_logger = WandbLogger(project="CSC8639_FewShot_FSS1000", run_name="/".join(Path(args.save_dir).parts[-2:]), config=vars(args))
    else:
        wandb_logger = None

    # Dataloader
    # mean, std = compute_mean_std(Path(args.img_dir))
    test_loader = DataLoader(
        EpisodeDataset(dataset_dir=args.dataset_dir,
                       n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query, episodes=args.pred_episodes, 
                       phase="test"), 
        batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=episodic_collate,
        pin_memory=True, worker_init_fn=worker_seed, persistent_workers=False)
    
    # Model
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

    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Save path
    save_dir = args.save_dir or find_latest_path(root_dir='result')
    os.makedirs(save_dir, exist_ok=True)

    # Visualizer
    visualizer = Visualizer(save_dir=save_dir,
                            max_save_samples=None, max_show_groups=0, wandb_logger=wandb_logger,
                            upload_pred=False, upload_overlay=False, upload_group=True)
    
    # Metric
    metric_fn = get_metric(names=args.metric, prefix='', per_image=False)
    total_metrics = {}

    # Predict
    with torch.no_grad():
        for step, (support_set, query_set, selected_classes) in enumerate(test_loader):
            support_set = [(img.to(device), mask.to(device), cls) for img, mask, cls in support_set]
            query_set = [(img.to(device), mask.to(device), cls) for img, mask, cls in query_set]
            query_imgs = torch.stack([img for img, _, _ in query_set])
            query_masks = torch.stack([mask for _, mask, _ in query_set])
            query_labels = [cls for _, _, cls in query_set]

            with torch.autocast(device_type=device, enabled=args.use_amp):
                outputs = model(support_set, query_set, selected_classes)  # model.forward(support_set, query_set, selected_classes)
                remapped_querymask = remap_querymask(query_masks, query_labels, selected_classes)
            
            preds = torch.argmax(outputs, dim=1)
            metrics = metric_fn(preds, remapped_querymask, list(range(len(selected_classes))))

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v.item()

            start_index = step * args.q_query

            query_preds = reverse_remap(preds, selected_classes)
            visualizer.save_group(query_imgs, query_masks, query_preds, start_index=start_index)

        # 只展示一组拼接图作为示例    
        visualizer.plot_demo(query_imgs, query_masks, preds)

    for k, v in total_metrics.items():
        total_metrics[k] = v / len(test_loader)
        print(f"{k}: {total_metrics[k]:.4f}")

    print(f"\nPrediction images saved to: {save_dir}")

    if args.wandb:
        wandb_logger.run.summary.update({f"{k}": v for k, v in total_metrics.items()})
        wandb_logger.finish()

if __name__ == '__main__':
    main()