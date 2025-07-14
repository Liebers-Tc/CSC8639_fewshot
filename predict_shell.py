import os
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from model.prototype_model import PrototypeNet
from model.prototype_resnet18_fix import PrototypeNet
from model.fewshot_unet import FewShotUNet
from model.prototype_resnet50 import ProtoSegNet
from model.prototype_resnet50_bg import ProtoResNet50BG
from utils.dataloader import EpisodeDataset
from utils.visualization import Visualizer
from utils.metrics import get_metric
from utils.savepath import find_latest_path
from utils.compute import compute_mean_std
from utils.mask_mapping import remap_querymask, reverse_remap


def parse_args():
    parser = argparse.ArgumentParser(description="FewShot Segmentation Prediction")

    parser.add_argument('--split_class_json', type=str, required=True)
    parser.add_argument('--class2images_mapping_json', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)

    parser.add_argument('--model_name', type=str, required=True)

    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--q_query', type=int, default=5)
    parser.add_argument('--pred_episodes', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--metric', nargs='+', default=['miou', 'dice', 'acc'])
    parser.add_argument('--ignore_index', type=int, default=None)

    parser.add_argument('--weight_path', type=str, required=True)
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
        wandb_logger = WandbLogger(project="CSC8639_FewShot", run_name="/".join(Path(args.save_dir).parts[-2:]), config=vars(args))
    else:
        wandb_logger = None

    # Dataloader
    # mean, std = compute_mean_std(Path(args.img_dir))
    test_loader = DataLoader(
        EpisodeDataset(split_class_json=args.split_class_json, 
                       class2images_mapping_json=args.class2images_mapping_json, 
                       img_dir=args.img_dir, 
                       mask_dir=args.mask_dir, 
                       n_way=args.n_way, k_shot=args.k_shot, q_query=args.q_query, episodes=args.pred_episodes, 
                       phase="test"), 
        batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=episodic_collate)
    
    # Model
    device = 'cuda'
    if args.model_name == 'prototype_resnet18':
        model = PrototypeNet(n_way=args.n_way).to(device)
    elif args.model_name == 'fewshot_unet':
        model = FewShotUNet(n_way=args.n_way).to(device)
    elif args.model_name == 'prototype_resnet18_fix':
        model = ProtoSegNet(n_way=args.n_way).to(device)
    elif args.model_name == 'prototype_resnet50':
        model = ProtoSegNet(n_way=args.n_way).to(device)
    elif args.model_name == 'prototype_resnet50_bg':
        model = ProtoResNet50BG(n_way=args.n_way).to(device)
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
                            max_save_samples=None, max_show_groups=3, wandb_logger=wandb_logger,
                            upload_pred=False, upload_overlay=False, upload_group=True)
    
    # Metric
    metric_fn = get_metric(names=args.metric, prefix='', per_image=False, ignore_index=args.ignore_index)
    total_metrics = {}

    # Predict
    with torch.no_grad():
        for step, (support_set, query_set, selected_classes) in enumerate(test_loader):
            support_set = [(img.to(device), mask.to(device), cls) for img, mask, cls in support_set]
            query_set = [(img.to(device), mask.to(device)) for img, mask in query_set]
            query_imgs = torch.stack([img for img, _ in query_set])
            query_masks = torch.stack([mask for _, mask in query_set])
            
            with torch.autocast(device_type=device, enabled=args.use_amp):
                outputs = model(support_set, query_set, selected_classes)  # model.forward(support_set, query_set, selected_classes)
                remapped_querymask = remap_querymask(query_masks, selected_classes, n_way=len(selected_classes))
            
            preds = torch.argmax(outputs, dim=1)
            # preds = reverse_remap(preds, selected_classes)
            # metrics = metric_fn(preds, query_masks, selected_classes)
            metrics = metric_fn(preds, remapped_querymask, list(range(len(selected_classes)+1)))

            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v.item()

            start_index = step * args.batch_size
            visualizer.save_group(query_imgs, query_masks, preds, start_index=start_index)

        # 只展示一组拼接图作为示例    
        visualizer.plot_demo(query_imgs, query_masks, preds)

    for k, v in total_metrics.items():
        total_metrics[k] = v / len(test_loader)
        print(f"{k}: {total_metrics[k]:.4f}")

    print(f"\nPrediction images saved to: {save_dir}")

    if args.wandb:
        wandb_logger.finish()

if __name__ == '__main__':
    main()