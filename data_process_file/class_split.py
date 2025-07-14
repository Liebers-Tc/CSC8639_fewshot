import random
import json
import os
from collections import defaultdict
from PIL import Image

random.seed(42)


""" ===构建 class2images 映射=== """
# 获取 mask 列表
mask_dir = "rawdata/mask"
mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]

class2images = defaultdict(list)
for fn in mask_files:
    mask = Image.open(os.path.join(mask_dir, fn))
    labels = set(mask.getdata())
    img_id = fn[:-4]
    for cls in labels:
        class2images[int(cls)].append(img_id)
class2images = dict(sorted(class2images.items(), key=lambda x: x[0]))

with open("class2images_mapping.json", "w") as f:
    json.dump(class2images, f, indent=2)
print("完成 class2images 映射")

print(f"各类映射图像数量：")
with open("class2images_number.txt", "w") as f:
    for cls, imgs in sorted(class2images.items(), key=lambda x: len(x[1])):
        print(f"类别 {cls}: {len(imgs)} 张图像")
        f.write(f"类别 {cls}: {len(imgs)} 张图像\n")




""" ===构建类别划分=== """
all_classes = list(class2images.keys())
# 移除背景类0 及 少于10张图像的类别
all_classes.remove(0)
all_classes = [cls for cls in all_classes if len(class2images[cls]) >= 10]

random.shuffle(all_classes)
n = len(all_classes)
train_classes = all_classes[:round(n * 0.85)]
val_classes   = all_classes[round(n * 0.85):round(n * 0.95)]
test_classes  = all_classes[round(n * 0.95):]

split = {
    "train_classes": train_classes,
    "val_classes": val_classes,
    "test_classes": test_classes
    }

with open("class_split.json", "w") as f:
    json.dump(split, f, indent=2)

print(f"\n完成 class 划分, 共 {n} 类")
print(f"train: {len(train_classes)} 类")
print(f"val: {len(val_classes)} 类")
print(f"test: {len(test_classes)} 类")