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


def fetch_squad(download=True):
    train_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    test_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    dataset_path = ROOTDIR / "squad"
    dataset_path.mkdir(exist_ok=True)
    datasets = []
    for url in (train_url, test_url):
        file_path = dataset_path / url.split("/")[-1]
        if file_path.exists():
            with file_path.open("r") as file:
                dataset = json.load(file)
        else:
            response = requests.get(url)
            assert response.status_code == HTTPStatus.OK
            dataset = json.loads(response.text)
            with file_path.open("w") as file:
                json.dump(dataset, file)
        datasets.append(dataset)
    return datasets
