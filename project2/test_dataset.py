import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from torchvision.transforms import v2
import sys
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

test_tf = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231]),
    ])


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith("png")]) 
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)


test_dataset = ImageDataset(root="/research/cvl-zeyuan/msu/cse849/project2/custom_image_dataset/test_unlabeled", transform=test_tf)
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)

print("test_dataset:", len(test_dataset))


for img, path in test_dataloader:
    print(img.shape, path)
    break