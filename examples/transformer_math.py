"""
https://github.com/tinygrad/tinygrad/blob/master/examples/transformer.py
"""
import random
import numpy as np
from tinygrad import Tensor, nn, TinyJit
from tinygrad.helpers import getenv, trange
from road_to_llm.models.transformer import Transformer

Tensor.manual_seed(0)


def make_dataset():
    ds = []
    for i in range(100):
        for j in range(100):
            s = i+j
            ds.append([i//10, i%10, j//10, j%10, s//100, (s//10)%10, s%10])
    random.shuffle(ds)
    ds = np.array(ds).astype(np.float32)
    ds_X = ds[:, 0:6]
    ds_Y = np.copy(ds[:, 1:])
    ds_X_train, ds_X_test = ds_X[0:8000], ds_X[8000:]
    ds_Y_train, ds_Y_test = ds_Y[0:8000], ds_Y[8000:]
    return ds_X_train, ds_Y_train, ds_X_test, ds_Y_test


@TinyJit
def train_step() -> Tensor:
    with Tensor.train():
        optim.zero_grad()
        samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
        output = model(Tensor(X_train)[samples])
        loss = output.sparse_categorical_crossentropy(Tensor(Y_train)[samples])
        loss.backward()
        optim.step()
        return loss


@TinyJit
def get_test_acc() -> np.float64:
    return (model(Tensor(X_test)).argmax(axis=-1) == Tensor(Y_test)).mean() * 100


model = Transformer(10, 6, 2, 128, 4, 32)
X_train, Y_train, X_test, Y_test = make_dataset()
lr = 0.003

test_acc = float("nan")
for _ in range(5):
    optim = nn.optim.Adam(nn.state.get_parameters(model), lr=lr)
    for _ in (t := trange(50)):
        loss = train_step()
        t.set_description(f"loss: {loss.item():6.2f}")
    print(f"test accuracy is {get_test_acc().item()}")
    lr /= 1.2
