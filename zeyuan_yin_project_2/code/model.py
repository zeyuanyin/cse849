import torch
import torch.nn as nn

torch.manual_seed(123)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(16) # BatchNorm layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # Conv 2 layer
        self.bn2 = nn.BatchNorm2d(32) # BatchNorm layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1) # Conv 3 layer
        self.bn3 = nn.BatchNorm2d(48) # BatchNorm layer
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1) # Conv 4 layer
        self.bn4 = nn.BatchNorm2d(64) # BatchNorm layer
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, stride=1, padding=1) # Conv 5 layer
        self.relu = nn.ReLU() # ReLU layer
        self.maxpool = nn.MaxPool2d(2) # MaxPool layer
        self.avgpool = nn.AdaptiveAvgPool2d(1) # Avgpool layer
        self.fc = nn.Linear(80, 10) # Linear layer
    
    def forward(self, x, intermediate_outputs=False):
        # TODO: Compute the forward pass output following the diagram in
        # the project PDF. If intermediate_outputs is True, return the
        # outputs of the convolutional layers as well.
        # print(x.shape) # torch.Size([32, 3, 40, 40])
        conv1_out = self.conv1(x)
        # print(conv1_out.shape) # torch.Size([32, 16, 40, 40])
        out = self.relu(self.bn1(conv1_out))
        conv2_out = self.conv2(out)
        out = self.relu(self.bn2(conv2_out))
        # print(out.shape) # torch.Size([32, 32, 40, 40])
        out = self.maxpool(out)
        # print(out.shape) # torch.Size([32, 32, 20, 20])
        conv3_out = self.conv3(out)
        out = self.relu(self.bn3(conv3_out))
        out = self.maxpool(out)
        # print(out.shape) # torch.Size([32, 48, 10, 10])
        conv4_out = self.conv4(out)
        out = self.relu(self.bn4(conv4_out))
        out = self.maxpool(out)
        # print(out.shape) # torch.Size([32, 64, 5, 5])
        conv5_out = self.conv5(out)
        # print(out.shape) # torch.Size([32, 64, 5, 5])
        out = self.avgpool(conv5_out)
        # print(out.shape) # torch.Size([32, 80, 1, 1])
        final_out = self.fc(out.squeeze())

        if intermediate_outputs:
            return final_out, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]
        else:
            return final_out
