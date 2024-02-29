import random
import numpy as np
from tqdm import trange
import torch
from torch import optim, nn
from road_to_llm.models.transformer import Transformer


def make_dataset(split_ratio=0.2):
    # https://github.com/tinygrad/tinygrad/blob/master/examples/transformer.py
    numbers = []
    for i in range(100):
        for j in range(100):
            s = i + j
            numbers.append([i // 10, i % 10, j // 10, j % 10, s // 100, (s // 10) % 10, s % 10])
    split_index = round(len(numbers) * (1 - split_ratio))
    random.shuffle(numbers)
    dataset = np.array(numbers).astype(np.float32)
    X = dataset[:, :-1]
    y = np.copy(dataset[:, 1:])
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:]


model = Transformer(vocab_size=10, hidden_dims=128, num_heads=4, num_layers=6, ff_dims=64, max_sequence_length=6, dropout=0.1)
X_train, y_train, X_test, y_test = make_dataset()
criterion = nn.CrossEntropyLoss()

num_epochs = 10
batch_size = 64
learning_rate = 0.003

for epoch in range(1, num_epochs+1):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for i in (t := trange((len(X_train) - 1) // batch_size + 1)):
        optimizer.zero_grad()
        sample = np.random.randint(0, len(X_train), size=batch_size)
        data = torch.tensor(X_train[sample], dtype=torch.long)
        label = torch.tensor(y_train[sample], dtype=torch.long)
        loss = criterion(model(data).transpose(1, 2), label)
        loss.backward()
        optimizer.step()
        t.set_description(f"{epoch=} iteration={i} loss={loss.item():.4f}")
    learning_rate *= 0.6
    model.eval()
    predictions = np.zeros(list(y_test.shape))
    for i in (t := trange((len(X_test) - 1) // batch_size + 1)):
        optimizer.zero_grad()
        data = torch.tensor(X_test[i*batch_size:(i+1)*batch_size], dtype=torch.long)
        predictions[i*batch_size:(i+1)*batch_size] = model(data).argmax(-1).detach().numpy()
    accuracy = (predictions == y_test).mean()
    print(f"test accuracy is {accuracy:.4f}")
