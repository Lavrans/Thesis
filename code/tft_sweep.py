from datetime import date
from math import isnan

from datasets import LSTMDataset
from models import TFT
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
from utils import get_tft_data

import wandb

criterion = nn.MSELoss()
device = torch_device('cuda' if cuda.is_available() else ('mps' if mps.is_available() else 'cpu'))
print(f'{device}')


# Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a
# model and return metric
def train_one_epoch(model: ScriptModule, train_loader: DataLoader, optimizer: Optimizer) -> float:
    train_loss = 0
    model.train()

    for idx, (r, s, t) in enumerate(train_loader):
        optimizer.zero_grad()
        pred, _, _ = model(s, r)
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
            pred, _, _ = model(s, r)
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
    d = wandb.config.d
    layers = wandb.config.layers
    dropout = wandb.config.dropout
    heads = wandb.config.heads
    timesteps = wandb.config.timesteps

    train_underlying, train_data, train_targets = get_tft_data(date(2020, 1, 1), date(2023, 1, 1), timesteps)
    val_underlying, val_data, val_targets = get_tft_data(date(2023, 1, 1), date(2023, 2, 1), timesteps)

    mlp_scaler = MinMaxScaler()
    train_data = mlp_scaler.fit_transform(train_data)
    val_data = mlp_scaler.transform(val_data)

    u_max = train_underlying.max()
    u_min = train_underlying.min()
    train_underlying = (train_underlying - u_min) / (u_max - u_min)
    val_underlying = (val_underlying - u_min) / (u_max - u_min)

    scaling_factor = train_targets.max() - train_targets.min()
    target_min = train_targets.min()

    train_targets = (train_targets - target_min) / scaling_factor
    val_targets = (val_targets - target_min) / scaling_factor

    train_dataset = LSTMDataset(train_data, train_underlying, train_targets, device)
    val_dataset = LSTMDataset(val_data, val_underlying, val_targets, device)

    model = TFT(d, 5, dropout, heads, layers)
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
                'train_loss': train_loss * scaling_factor**2,
                'val_loss': val_loss * scaling_factor**2,
            }
        )
        if val_loss >= min_val_loss:
            counter += 1
        else:
            counter = 0
            min_val_loss = val_loss

        scheduler.step()
        epoch += 1


if __name__ == '__main__':
    main()
