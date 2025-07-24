from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T


def compute_mean_std(image_dir):
    """
    计算指定图像文件夹中的所有图片的均值和标准差（以 Channel 为单位）
    """

    image_paths = sorted(list(Path(image_dir).glob('*.jpg')))
    pixel_sum = torch.zeros(3)
    pixel_squared_sum = torch.zeros(3)
    num_pixels = 0

    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = T.ToTensor()(img)
        c, h, w = img.shape
        pixel_sum += img.sum(dim=(1, 2))
        pixel_squared_sum += (img ** 2).sum(dim=(1, 2))
        num_pixels += h * w

    mean = (pixel_sum / num_pixels).tolist()
    std = ((pixel_squared_sum / num_pixels - torch.tensor(mean) ** 2) ** 0.5).tolist()
    return mean, std