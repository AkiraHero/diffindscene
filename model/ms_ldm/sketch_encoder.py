import torch
import torch.nn as nn


class SketchEncoder(nn.Module):
    def __init__(self, out_chn=768) -> None:
        super().__init__()
        self.out_chn = out_chn
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, out_chn, kernel_size=2, stride=2, padding=0)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(bs, self.out_chn, -1).permute(0, 2, 1)
        return x
