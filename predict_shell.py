import os
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from model.unet_model import UNet
from model.prototype_model import PrototypeNet
from utils.dataloader import EpisodeDataset
from utils.visualization import Visualizer
from utils.metrics import get_metric
from utils.savepath import find_latest_path
from utils.compute import compute_mean_std


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation Prediction")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=104)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--backbone', type=str)

    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--wandb', action='store_true')

    return parser.parse_args()


def main():
    args = parse_args()

    # init wandb_logger
    if args.wandb:
        from utils.wandb_utils import WandbLogger
        wandb_logger = WandbLogger(project="CSC8639", run_name=args.save_dir, config=vars(args))
    else:
        wandb_logger = None

    # Dataloader
    mean, std = compute_mean_std(Path(args.dataset) / 'train' / 'images')
    test_loader = DataLoader(
        EpisodeDataset(split_class_json="./data/fs_dataset_256/class_split.json", 
                       class2images_mapping_json="./data/fs_dataset_256/class2images_mapping.json", 
                       img_dir="./data/fs_dataset_256/rawdata/image", 
                       mask_dir="./data/fs_dataset_256/rawdata/mask", 
                       n_way=5, k_shot=5, q_query=5, phase="val"), 
        batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model_name == 'unet':
        model = UNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    elif args.model_name == 'proto':
        model = PrototypeNet(in_channels=args.in_channels, num_classes=args.num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    checkpoint = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Save path
    save_dir = args.save_dir or find_latest_path(root_dir='result')
    os.makedirs(save_dir, exist_ok=True)

    # Visualizer
    visualizer = Visualizer(save_dir=save_dir, mean=mean, std=std,
                            max_save_samples=None, max_show_groups=3, wandb_logger=wandb_logger,
                            upload_pred=False, upload_overlay=False, upload_group=True)
    
    # Metric
    metric_fn = get_metric(names=['miou', 'dice', 'acc'], num_classes=args.num_classes, per_image=False)
    total_metrics = {}

    # Predict
    with torch.no_grad():
        for step, (support_set, query_set) in enumerate(test_loader):
            support_set = [(img.to(device), mask.to(device)) for img, mask in support_set]
            query_imgs = torch.stack([img for img, _ in query_set]).to(device)
            query_masks = torch.stack([mask for _, mask in query_set]).to(device)
            
            with torch.autocast(device_type=device, enabled=args.use_amp):
                outputs = model(support_set, query_set)

            preds = torch.argmax(outputs, dim=1)
            start_index = step * args.batch_size
            visualizer.save_group(query_imgs, query_masks, preds, start_index=start_index)

            batch_metrics = metric_fn(outputs, query_masks)
            for k, v in batch_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v.item()
        visualizer.plot_demo(query_imgs, query_masks, preds)

    for k, v in total_metrics.items():
        total_metrics[k] = v / len(test_loader)
        print(f"{k}: {total_metrics[k]:.4f}")

    print(f"\nPrediction images saved to: {save_dir}")

    if args.wandb:
        wandb_logger.finish()

if __name__ == '__main__':
    main()