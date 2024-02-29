from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from road_to_llm.common.helpers import get_gpu
from road_to_llm.common.dataloader import fetch_cifar10
from road_to_llm.models.resnet import resnet50

torch.manual_seed(42)
device = get_gpu()

num_epochs = 10
batch_size = 32
learning_rate = 0.005
weight_decay = learning_rate / num_epochs

train_dataset, test_dataset = fetch_cifar10()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = resnet50(num_classes=10, zero_init_residual=True)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs + 1):
    model.train()
    for batch, (data, target) in (t := tqdm(enumerate(train_loader), total=len(train_loader))):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        t.set_description(f"{epoch=} {batch=} loss={loss.item():.4f}")
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        predictions = model(data).argmax(-1)
        correct += (predictions == target).sum().item()
    print(f"test accuracy = {correct / len(test_dataset):.4f}\n")
