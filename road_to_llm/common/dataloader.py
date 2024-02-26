from pathlib import Path
import torch
from torchvision import datasets, transforms

ROOTDIR = Path(__file__).parent.parent.parent / "datasets"


def fetch_mnist(download=True):
    transform = transforms.Compose([transforms.PILToTensor(), lambda x: x / 255.0])
    train = datasets.MNIST(ROOTDIR, train=True, transform=transform, target_transform=torch.tensor, download=download)
    test = datasets.MNIST(ROOTDIR, train=False, transform=transform, target_transform=torch.tensor, download=download)
    return train, test


def fetch_cifar10(download=True, size=256, crop=224):
    transform = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(crop), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train = datasets.CIFAR10(ROOTDIR, train=True, transform=transform, target_transform=torch.tensor, download=download)
    test = datasets.CIFAR10(ROOTDIR, train=False, transform=transform, target_transform=torch.tensor, download=download)
    return train, test
