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
model.to(device)

for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
# TODO: Initialize weights. You may use kaiming_normal_ for
# initialization. Check this StackOverflow answer:
# https://stackoverflow.com/a/49433937/6211109

# Set your training parameters here
num_epochs = 100
# num_epochs = 1
lr = 0.005
weight_decay = 0.0001


# png_path = "q1_plots_W_bs32_lr0.01_Erasing_noise.png"

# png_path = "q1_plots_W_bs32_lr0.005_Erasing_onecycle_0.3_25_10000_noise_s0.005.png"

png_path = "q1_plots_W_bs32_lr0.005_Erasing_onecycle_0.3_25_10000_E100.png"
model_path = "q1_plots_W_bs32_lr0.005_Erasing_onecycle_0.3_25_10000_E100.pt"

# png_path = "q1_plots_W_bs32_lr0.005_Erasing_onecycle_0.3_25_10000_noise_s0.005_E300.png"
# model_path = "q1_plots_W_bs32_lr0.005_Erasing_onecycle_0.3_25_10000_noise_s0.005_E300.pt"


# png_path = "q1_plots_W_bs32_lr0.005_onecycle_0.3_25_10000_E300.png"
# model_path = "q1_plots_W_bs32_lr0.005_onecycle_0.3_25_10000_E300.pt"

# Setup your cross-entropy loss function
loss_fn = nn.CrossEntropyLoss()

# Setup your optimizer that uses lr and weight_decay
optimizer =  torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)



# Setup your learning rate scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr/100)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                                steps_per_epoch=len(train_loader), 
                                                epochs=num_epochs,
                                                # total_steps=num_epochs*len(train_loader), 
                                                pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=10000)

# For plotting.
step = 0
train_step_list = []
train_loss_list = []
train_accuracy_list = []
val_step_list = []
val_loss_list = []
val_accuracy_list = []

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        # TODO: Move images and labels to device
        # TODO: Zero the gradients
        # TODO: Forward pass through the model
        # TODO: Calculate the loss
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # TODO: Backward pass
        # TODO: Update weights
        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())
        train_step_list.append(step)

        step += 1
        
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step() # Step with the scheduler
    
    model.eval()
    with torch.no_grad():
        # Compute validation loss and accuracy
        correct, total = 0, 0
        avg_loss = 0.
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

        val_loss_list.append(avg_loss)
        val_accuracy_list.append(val_accuracy)
        train_accuracy_list.append(train_accuracy)
        val_step_list.append(step)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Val acc: {val_accuracy:.2f}%,",
              f"Train acc: {train_accuracy:.2f}%")
        
        # Optionally, you can save only your best model so far by
        # keeping track of best validation accuracies.
        torch.save(model.state_dict(), model_path)

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(train_step_list, train_loss_list, label="Train")
        axs[0].plot(val_step_list, val_loss_list, label="Val")
        axs[0].set_yscale("log")

        axs[1].plot(val_step_list, train_accuracy_list, label="Train")
        axs[1].plot(val_step_list, val_accuracy_list, label="Val")

        axs[0].set_title("Loss")
        axs[1].set_title("Accuracy")

        for ax in axs:
            ax.legend()
            ax.grid()
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")

        plt.tight_layout()
        # plt.savefig(f"q1_plots_W.png", dpi=300)
        
        plt.savefig(png_path, dpi=300)
        plt.clf()
        plt.close()

torch.save(model.state_dict(), "q1_model.pt")

# You can copy-paste the following code to another program to evaluate
# your model separately.
model.load_state_dict(torch.load("q1_model.pt", weights_only=True))
model.eval()
test_images = sorted(glob("custom_image_dataset/test_unlabeled/*.png"))

# TODO: Create test-time image transformations. Same as what you used
# for validation.
test_tf = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231]),
    ])

test_write = open("q1_test.txt", "w")
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

