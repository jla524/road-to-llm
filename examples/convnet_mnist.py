from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from road_to_llm.common.helpers import get_gpu
from road_to_llm.common.dataloader import fetch_mnist
from road_to_llm.models.convnet import ConvNet

torch.manual_seed(42)
device = get_gpu()

num_epochs = 10
batch_size = 128
learning_rate = 0.005

train_dataset, test_dataset = fetch_mnist()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ConvNet()
model.to(device)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs + 1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for batch, (data, target) in (t := tqdm(enumerate(train_loader))):
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
    learning_rate *= 0.7
    print(f"test accuracy = {correct / len(test_dataset):.4f}\n")
