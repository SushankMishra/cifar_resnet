import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return x + self.block(x)


class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1_conv = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer1_res = ResBlock(128)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer3_conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer3_res = ResBlock(512)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.prep(x)
        x1 = self.layer1_conv(out)
        out = x1 + self.layer1_res(x1)
        out = self.layer2(out)
        x3 = self.layer3_conv(out)
        out = x3 + self.layer3_res(x3)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)


def get_custom_resnet(num_classes=10):
    return CustomResNet(num_classes=num_classes)