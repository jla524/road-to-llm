import json
from pathlib import Path
from http import HTTPStatus
from io import StringIO
import torch
import requests
import pandas as pd
from torch.utils.data.distributed import Dataset
from torchvision import datasets, transforms

ROOTDIR = Path(__file__).parent.parent.parent / "extra" / "datasets"
ROOTDIR.mkdir(parents=True, exist_ok=True)


def fetch_mnist(download=True):
    transform = transforms.Compose([transforms.PILToTensor(), lambda x: x / 255.0])
    kwargs = {"transform": transform, "target_transform": torch.tensor, "download": download}
    train = datasets.MNIST(ROOTDIR, train=True, **kwargs)
    test = datasets.MNIST(ROOTDIR, train=False, **kwargs)
    return train, test


def fetch_cifar10(download=True, size=256, crop=224):
    transform = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(crop), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    kwargs = {"transform": transform, "target_transform": torch.tensor, "download": download}
    train = datasets.CIFAR10(ROOTDIR, train=True, **kwargs)
    test = datasets.CIFAR10(ROOTDIR, train=False, **kwargs)
    return train, test


def fetch_spamdata(download=True):
    url = "https://raw.githubusercontent.com/prateekjoshi565/Fine-Tuning-BERT/master/spamdata_v2.csv"
    file_path = ROOTDIR / url.split("/")[-1]
    if file_path.exists():
        df = pd.read_csv(file_path)
    else:
        response = requests.get(url)
        assert response.status_code == HTTPStatus.OK
        df = pd.read_csv(StringIO(response.text))
        df.to_csv(file_path, index=False)
    return df
