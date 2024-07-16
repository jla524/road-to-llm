from tinygrad import Tensor, nn, TinyJit
from tinygrad.helpers import getenv, trange
from road_to_llm.models.mlp import MLP

Tensor.manual_seed(42)

X_train, Y_train, X_test, Y_test = nn.datasets.mnist()
model = MLP()
opt = nn.optim.Adam(nn.state.get_parameters(model))

@TinyJit
def train_step() -> Tensor:
    with Tensor.train():
        opt.zero_grad()
        samples = Tensor.randint(getenv("BS", 512), high=X_train.shape[0])
        output = model(X_train[samples])
        loss = output.sparse_categorical_crossentropy(Y_train[samples])
        loss.backward()
        opt.step()
        return loss


@TinyJit
def get_test_acc() -> Tensor:
    return (model(X_test).argmax(axis=1) == Y_test).mean() * 100

test_acc = float("nan")
for i in (t := trange(1000)):
    loss = train_step()
    if i % 10 == 9: test_acc = get_test_acc().item()
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")
