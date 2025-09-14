# CIFAR10 Trainer

A modular PyTorch package for training image classification models on CIFAR-10.  
Supports:
- Custom ResNet-like architecture (with residual blocks, prep layers, and skip connections).
- Albumentations-based data augmentation (Cutout included).
- Learning Rate Finder (`torch-lr-finder`).
- One Cycle Policy training.

---

## 🚀 Features
- **Custom Architecture**: Residual network with multiple blocks and skip connections.
- **Albumentations Augmentations**: Powerful augmentations including Cutout.
- **Learning Rate Finder**: Automatically suggest best LR range.
- **One Cycle Policy**: Efficient training schedule.
---

Usage

Training
python -m cifar10_trainer.main --epochs 24 --batch-size 128 --lr-finder