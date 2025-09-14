import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cifar_resnet.models.resnet_cifar import get_custom_resnet
from cifar_resnet.data.datasets import CIFAR10Alb, get_transforms
from cifar_resnet.utils.train_utils import train_one_epoch, eval_one_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_ds = CIFAR10Alb(train=True, transform=get_transforms(train=True))
test_ds = CIFAR10Alb(train=False, transform=get_transforms(train=False))
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# Model, Optimizer, Scheduler
model = get_custom_resnet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 24
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=3e-3,      # to be tuned using lr_finder
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=5/24,
    anneal_strategy='cos',
    div_factor=10,
    final_div_factor=100
)

# Training Loop
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion, scheduler)
    test_loss, test_acc = eval_one_epoch(model, device, test_loader, criterion)
    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")