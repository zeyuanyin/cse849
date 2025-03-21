import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

torch.manual_seed(123)


def create_dataloaders():
    train_tf = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231]),
        v2.GaussianNoise(mean=0.432, sigma=0.001, clip=False),
        v2.RandomErasing(scale=(0.02, 0.33)),
    ])
    val_tf = v2.Compose([
        # v2.ToTensor(),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231]),
    ])
    train_dataset = ImageFolder(root='/research/cvl-zeyuan/msu/cse849/project2/custom_image_dataset/train', transform=train_tf)
    val_dataset = ImageFolder(root='/research/cvl-zeyuan/msu/cse849/project2/custom_image_dataset/val', transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=8)

    print('train_dataset:', len(train_dataset))
    print('val_dataset:', len(val_dataset))
    return train_loader, val_loader

