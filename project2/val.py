import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.transforms import v2

torch.manual_seed(123)

from data import create_dataloaders
from model import CNN
from torch.utils.data import Dataset


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get your dataloaders
train_loader, val_loader = create_dataloaders()

# Get your model
model = CNN()
# load the weights
model.load_state_dict(torch.load("q1_plots_W_bs32_lr0.005_Erasing_onecycle_0.3_25_10000_E300.pt", weights_only=True))

model.to(device)

loss_fn = nn.CrossEntropyLoss()

test_write = open("q1_test_E300.txt", "w")

# For plotting.
step = 0
train_step_list = []
train_loss_list = []
train_accuracy_list = []
val_step_list = []
val_loss_list = []
val_accuracy_list = []


model.eval()
with torch.no_grad():
    # Compute validation loss and accuracy
    correct, total = 0, 0
    avg_loss = 0.0
    for images, labels in val_loader:
        # TODO: Forward pass similar to training
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        avg_loss += loss.item() * labels.size(0)
        # TODO: Get the predicted labels from the model's outputs
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_accuracy = correct / total * 100
    avg_loss /= total

    # Similarly compute training accuracy. This training accuracy is
    # not fully reliable as the image transformations are different
    # from the validation transformations. But it will inform you of
    # potential issues.
    correct, total = 0, 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total * 100

    print(f"Val acc: {val_accuracy:.2f}%,", f"Train acc: {train_accuracy:.2f}%")


# You can copy-paste the following code to another program to evaluate
# your model separately.
model.load_state_dict(torch.load("q1_model.pt", weights_only=True))
model.eval()
test_images = sorted(glob("custom_image_dataset/test_unlabeled/*.png"))

# TODO: Create test-time image transformations. Same as what you used
# for validation.
test_tf = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231]),
    ]
)

# We will run through each image and write the predictions to a file.
# You may also write a custom Dataset class to load it more efficiently.


test_dataset = ImageDataset(root="/research/cvl-zeyuan/msu/cse849/project2/custom_image_dataset/test_unlabeled", transform=test_tf)
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)


print("test_dataset:", len(test_dataset))


# for imgfile in test_images:

for img, path in test_dataloader:
    img = img.to(device)
    # TODO: Forward pass through the model and get the predicted label
    outputs = model(img)
    predicted = torch.argmax(outputs, dim=1)

    # predicted is a PyTorch tensor containing the predicted label as a
    # single value between 0 and 9 (inclusive)
    for filename, pred in zip(path, predicted):
        # print(filename, pred)
        test_write.write(f"{filename},{pred}\n")
test_write.close()
