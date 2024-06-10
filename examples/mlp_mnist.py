from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from road_to_llm.common.dataloader import fetch_mnist
from road_to_llm.models.mlp import MLP

torch.manual_seed(42)

num_epochs = 10
batch_size = 128
learning_rate = 0.005

train_dataset, test_dataset = fetch_mnist()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MLP(784, 10)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs + 1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for batch, (data, target) in (t := tqdm(enumerate(train_loader))):
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        t.set_description(f"{epoch=} {batch=} loss={loss.item():.4f}")
    model.eval()
    correct = 0
    for data, target in test_loader:
        predictions = model(data).argmax(-1)
        correct += (predictions == target).sum().item()
    learning_rate *= 0.7  # yields higher accuracy than weight decay!
    print(f"test accuracy = {correct / len(test_dataset):.4f}\n")
