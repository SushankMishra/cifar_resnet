from setuptools import setup, find_packages

setup(
    name="cifar10_trainer",
    version="0.1.0",
    description="Modular CIFAR-10 training package with ResNet, Albumentations, LR Finder, and OneCycleLR",
    author="Sushank Mishra",
    author_email="sushankmishraiitd@gmail.com",
    url="",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "albumentations>=1.3.0",
        "numpy>=1.23.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "torch-lr-finder>=0.2.1",
        "opencv-python>=4.7.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)