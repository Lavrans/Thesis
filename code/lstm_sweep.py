from datetime import date
from math import isnan

from datasets import LSTMDataset
from models import LstmMLP
from numpy import inf
from sklearn.preprocessing import MinMaxScaler
from torch import cuda
from torch import device as torch_device
from torch import nn
from torch import no_grad
from torch.backends import mps
from torch.jit import ScriptModule
from torch.optim import Adam
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from utils import get_lstm_data

import wandb

# Define sweep config
# sweep_configuration = {
#     'program': 'code/lstm_sweep.py',
#     'method': 'bayes',
#     'name': 'sweep',
#     'metric': {'goal': 'minimize', 'name': 'val_loss.min'},
#     'parameters': {
#         'batch_size': {'values': [1024, 2048, 4096]},
#         'lr': {'max': 0.1, 'min': 0.0001},
#         'gamma': {'max': 1.0, 'min': 0.80},
#     },
# }

# Initialize sweep by passing in config.
# (Optional) Provide a name of the project.
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='lstm-sweep')

criterion = nn.MSELoss()
device = torch_device('cuda' if cuda.is_available() else ('mps' if mps.is_available() else 'cpu'))
print(f'{device}')

train_underlying, train_data, train_targets = get_lstm_data(date(2012, 1, 1), date(2015, 1, 1))
val_underlying, val_data, val_targets = get_lstm_data(date(2015, 1, 1), date(2015, 2, 1))

mlp_scaler = MinMaxScaler()
train_data = mlp_scaler.fit_transform(train_data)
val_data = mlp_scaler.transform(val_data)

u_max = train_underlying.max()
u_min = train_underlying.min()
train_underlying = (train_underlying - u_min) / (u_max - u_min)
val_underlying = (val_underlying - u_min) / (u_max - u_min)

# target_scaler = MinMaxScaler()
# train_data[['C_PRICE']] = target_scaler.fit_transform(train_data[['C_PRICE']])
# val_data[['C_PRICE']] = target_scaler.transform(val_data[['C_PRICE']])

train_dataset = LSTMDataset(train_data, train_underlying, train_targets, device)
val_dataset = LSTMDataset(val_data, val_underlying, val_targets, device)


# Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a
# model and return metric
def train_one_epoch(model: ScriptModule, train_loader: DataLoader, optimizer: Optimizer) -> float:
    train_loss = 0
    model.train()

    for idx, (r, s, t) in enumerate(train_loader):
        optimizer.zero_grad()
        x = (r, s)
        pred = model(x)
        loss = criterion(pred.squeeze(), t)

        item = loss.item()
        if not isnan(item):
            train_loss += item

            loss.backward()
            optimizer.step()

    mean_loss_train = train_loss / (idx + 1)
    return mean_loss_train


def evaluate_one_epoch(model: ScriptModule, val_loader: DataLoader) -> float:
    val_loss = 0
    model.eval()

    with no_grad():
        for idx, (r, s, t) in enumerate(val_loader):
            x = (r, s)
            pred = model(x)
            loss = criterion(pred.squeeze(), t)
            val_loss += loss.item()
    mean_loss_val = val_loss / (idx + 1)
    return mean_loss_val


def main() -> None:
    wandb.init()
    wandb.define_metric('val_loss', summary='min,max,mean,last')

    # note that we define values from `wandb.config`
    # instead of defining hard values
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    gamma = wandb.config.gamma
    model = LstmMLP()
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=gamma)

    train_loader = DataLoader(train_dataset, bs, True)
    val_loader = DataLoader(val_dataset, bs, False)

    tolerance = 20
    counter = 0
    epoch = 1
    min_val_loss = inf

    while counter < tolerance:
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_loss = evaluate_one_epoch(model, val_loader)

        wandb.log(
            {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
        )
        if val_loss >= min_val_loss:
            counter += 1
        else:
            counter = 0
            min_val_loss = val_loss

        scheduler.step()
        epoch += 1


# wandb.agent(sweep_id, function=main, count=75)

if __name__ == '__main__':
    main()
