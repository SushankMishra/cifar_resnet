import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(train=True):
    if train:
        return A.Compose([
            A.PadIfNeeded(min_height=36, min_width=36, border_mode=0, value=[0,0,0]),
            A.RandomCrop(32, 32),
            A.HorizontalFlip(p=0.5),
            A.CoarseDropout(
                max_holes=1, max_height=8, max_width=8,
                min_holes=1, min_height=8, min_width=8,
                fill_value=(int(0.4914*255), int(0.4822*255), int(0.4465*255)), p=0.5
            ),
            A.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.247, 0.243, 0.261)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.247, 0.243, 0.261)),
            ToTensorV2(),
        ])


class CIFAR10Alb(Dataset):
    def __init__(self, root="./data", train=True, transform=None, download=True):
        self.ds = CIFAR10(root=root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        img = np.array(img)  # albumentations needs numpy
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label