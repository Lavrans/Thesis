from copy import deepcopy
from datetime import date
from math import isnan
from typing import Tuple

from datasets import LSTMDataset
from datasets import MLPDataset
from dateutil.relativedelta import relativedelta
from models import BS
from models import MLP
from models import TFT
from models import LstmMLP
from numpy import inf
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch import concat
from torch import cuda
from torch import device as torch_device
from torch import nn
from torch import no_grad
from torch import save
from torch import stack
from torch import tensor
from torch.backends import mps
from torch.jit import ScriptModule
from torch.nn import Module
from torch.optim import Adam
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from utils import get_bs_data
from utils import get_lstm_data
from utils import get_mlp_data
from utils import get_tft_data

import wandb

criterion = nn.MSELoss()
device = torch_device('cuda' if cuda.is_available() else ('mps' if mps.is_available() else 'cpu'))
print(f'{device}')


def get_dates(offset_months: int, offset_years: int):
    train_from = date(2011, 1, 1) + relativedelta(months=offset_months, years=offset_years)
    train_to = train_from + relativedelta(years=3)
    val_to = train_to + relativedelta(months=1)
    test_to = val_to + relativedelta(months=1)
    return {
        'train': (train_from, train_to),
        'val': (train_to, val_to),
        'test': (val_to, test_to),
    }


# Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a
# model and return metric
def train_one_epoch_tft(model: ScriptModule, train_loader: DataLoader, optimizer: Optimizer) -> float:
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


def train_one_epoch_mlp(model: Module, train_loader: DataLoader, optimizer: Optimizer) -> float:
    train_loss = 0
    model.train()

    for idx, (s, t) in enumerate(train_loader):
        optimizer.zero_grad()
        x = s.float().to(device)
        pred = model(x)
        loss = criterion(pred.squeeze(), t.float().to(device))

        item = loss.item()

        if not isnan(item):
            train_loss += item

            loss.backward()
            optimizer.step()

    mean_loss_train = train_loss / (idx + 1)
    return mean_loss_train


def train_one_epoch_lstm(model: ScriptModule, train_loader: DataLoader, optimizer: Optimizer) -> float:
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


def evaluate_one_epoch_tft(model: ScriptModule, val_loader: DataLoader) -> float:
    val_loss = 0
    model.eval()

    with no_grad():
        for idx, (r, s, t) in enumerate(val_loader):
            pred, _, _ = model(s, r)
            loss = criterion(pred.squeeze(), t)
            val_loss += loss.item()
    mean_loss_val = val_loss / (idx + 1)
    return mean_loss_val


def evaluate_one_epoch_mlp(model: Module, val_loader: DataLoader) -> float:
    val_loss = 0
    model.eval()

    with no_grad():
        for idx, (s, t) in enumerate(val_loader):
            x = s.float().to(device)
            pred = model(x)
            loss = criterion(pred.squeeze(), t.float().to(device))
            val_loss += loss.item()
    mean_loss_val = val_loss / (idx + 1)
    return mean_loss_val


def evaluate_one_epoch_lstm(model: ScriptModule, val_loader: DataLoader) -> float:
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


def test_tft(model: Module, test_loader: DataLoader, scaling_factor, target_min) -> Tuple[float, Tensor]:
    test_loss = 0
    model.eval()
    preds = []

    with no_grad():
        for idx, (r, s, t) in enumerate(test_loader):
            pred, _, _ = model(s, r)
            pred = pred.squeeze() * scaling_factor + target_min
            loss = criterion(pred, t)
            test_loss += loss.item()
            preds.append(pred)
    mean_loss_val = test_loss / (idx + 1)
    return mean_loss_val, concat(preds)


def test_mlp(model: Module, test_loader: DataLoader, scaling_factor, target_min) -> Tuple[float, Tensor]:
    val_loss = 0
    model.eval()
    preds = []

    with no_grad():
        for idx, (s, t) in enumerate(test_loader):
            x = s.float().to(device)
            pred = model(x)
            pred = pred.squeeze() * scaling_factor + target_min
            loss = criterion(pred, t.float().to(device))
            val_loss += loss.item()
            preds.append(pred)
    mean_loss_val = val_loss / (idx + 1)
    return mean_loss_val, concat(preds)


def test_lstm(model: ScriptModule, val_loader: DataLoader, scaling_factor, target_min) -> Tuple[float, Tensor]:
    val_loss = 0
    model.eval()
    preds = []

    with no_grad():
        for idx, (r, s, t) in enumerate(val_loader):
            x = (r, s)
            pred = model(x)
            pred = pred.squeeze() * scaling_factor + target_min
            loss = criterion(pred, t)
            val_loss += loss.item()
            preds.append(pred)
    mean_loss_val = val_loss / (idx + 1)
    return mean_loss_val, concat(preds)


def get_lstm_dataloaders(from_dates, to_dates, timesteps):
    train_underlying, train_data, train_targets = get_lstm_data(from_dates[0], to_dates[0], timesteps)
    val_underlying, val_data, val_targets = get_lstm_data(from_dates[1], to_dates[1], timesteps)
    test_underlying, test_data, test_targets = get_lstm_data(from_dates[2], to_dates[2], timesteps)

    mlp_scaler = MinMaxScaler()
    train_data = mlp_scaler.fit_transform(train_data)
    val_data = mlp_scaler.transform(val_data)
    test_data = mlp_scaler.transform(test_data)

    u_max = train_underlying.max()
    u_min = train_underlying.min()
    train_underlying = (train_underlying - u_min) / (u_max - u_min)
    val_underlying = (val_underlying - u_min) / (u_max - u_min)
    test_underlying = (test_underlying - u_min) / (u_max - u_min)

    target_scaler = MinMaxScaler()
    train_targets = target_scaler.fit_transform(train_targets.reshape(-1, 1)).squeeze()
    val_targets = target_scaler.transform(val_targets.reshape(-1, 1)).squeeze()

    return (
        DataLoader(LSTMDataset(train_data, train_underlying, train_targets, device), 4096, True),
        DataLoader(LSTMDataset(val_data, val_underlying, val_targets, device), 4096, False),
        DataLoader(LSTMDataset(test_data, test_underlying, test_targets, device), 4096, False),
    )


def get_tft_dataloaders(from_dates, to_dates, timesteps):
    train_underlying, train_data, train_targets = get_tft_data(from_dates[0], to_dates[0], timesteps)
    val_underlying, val_data, val_targets = get_tft_data(from_dates[1], to_dates[1], timesteps)
    test_underlying, test_data, test_targets = get_tft_data(from_dates[2], to_dates[2], timesteps)

    mlp_scaler = MinMaxScaler()
    train_data = mlp_scaler.fit_transform(train_data)
    val_data = mlp_scaler.transform(val_data)
    test_data = mlp_scaler.transform(test_data)

    u_max = train_underlying.max()
    u_min = train_underlying.min()
    train_underlying = (train_underlying - u_min) / (u_max - u_min)
    val_underlying = (val_underlying - u_min) / (u_max - u_min)
    test_underlying = (test_underlying - u_min) / (u_max - u_min)

    target_scaler = MinMaxScaler()
    train_targets = target_scaler.fit_transform(train_targets.reshape(-1, 1)).squeeze()
    val_targets = target_scaler.transform(val_targets.reshape(-1, 1)).squeeze()

    return (
        DataLoader(LSTMDataset(train_data, train_underlying, train_targets, device), 4096, True),
        DataLoader(LSTMDataset(val_data, val_underlying, val_targets, device), 4096, False),
        DataLoader(LSTMDataset(test_data, test_underlying, test_targets, device), 4096, False),
    )


def main() -> None:
    wandb.init()

    tft = TFT(128, 5, 0.2, 1, 3)
    tft.to(device)
    tft_optimizer = Adam(tft.parameters(), lr=0.001)
    tft_scheduler = ExponentialLR(optimizer=tft_optimizer, gamma=0.975)

    lstm = LstmMLP()
    lstm.to(device)
    lstm_optimizer = Adam(lstm.parameters(), lr=0.057)
    lstm_scheduler = ExponentialLR(optimizer=lstm_optimizer, gamma=0.91)

    offset_months = wandb.config.offset_months
    offset_years = wandb.config.offset_years

    dates = get_dates(offset_months, offset_years)

    train_from, train_to = dates['train']
    val_from, val_to = dates['val']
    test_from, test_to = dates['test']

    mlp_train_data, mlp_train_targets = get_mlp_data(train_from, train_to)
    mlp_val_data, mlp_val_targets = get_mlp_data(val_from, val_to)
    mlp_test_data, mlp_test_targets = get_mlp_data(test_from, test_to)

    mlp_scaler = MinMaxScaler()
    mlp_train_data = mlp_scaler.fit_transform(mlp_train_data)
    mlp_val_data = mlp_scaler.transform(mlp_val_data)
    mlp_test_data = mlp_scaler.transform(mlp_test_data)

    scaling_factor = mlp_train_targets.max() - mlp_train_targets.min()
    target_min = mlp_train_targets.min()

    mlp_train_targets = (mlp_train_targets - target_min) / scaling_factor
    mlp_val_targets = (mlp_val_targets - target_min) / scaling_factor

    mlp_train_dataloader = DataLoader(MLPDataset(mlp_train_data, mlp_train_targets, device), 4096, True)
    mlp_val_dataloader = DataLoader(MLPDataset(mlp_val_data, mlp_val_targets, device), 4096, False)
    mlp_test_dataloader = DataLoader(MLPDataset(mlp_test_data, mlp_test_targets, device), 4096, False)

    from_dates = [train_from, val_from, test_from]
    to_dates = [train_to, val_to, test_to]

    lstm_train_loader, lstm_val_loader, lstm_test_loader = get_lstm_dataloaders(from_dates, to_dates, 140)
    tft_train_loader, tft_val_loader, tft_test_loader = get_tft_dataloaders(from_dates, to_dates, 240)

    mlp = MLP(5, 200, 0.094)
    mlp.to(device)
    mlp_optimizer = Adam(mlp.parameters(), lr=0.004023)
    mlp_scheduler = ExponentialLR(optimizer=mlp_optimizer, gamma=0.9404)

    tolerance = 20
    counter = 0
    min_val_loss = inf

    while counter < tolerance:
        train_one_epoch_mlp(mlp, mlp_train_dataloader, mlp_optimizer)
        val_loss = evaluate_one_epoch_mlp(mlp, mlp_val_dataloader)
        if val_loss >= min_val_loss:
            counter += 1
        else:
            counter = 0
            min_val_loss = val_loss
            mlp_copy = deepcopy(mlp.state_dict())

        mlp_scheduler.step()

    tolerance = 20
    counter = 0
    min_val_loss = inf

    while counter < tolerance:
        train_one_epoch_lstm(lstm, lstm_train_loader, lstm_optimizer)
        val_loss = evaluate_one_epoch_lstm(lstm, lstm_val_loader)
        if val_loss >= min_val_loss:
            counter += 1
        else:
            counter = 0
            min_val_loss = val_loss
            lstm_copy = deepcopy(lstm.state_dict())

        lstm_scheduler.step()

    tolerance = 20
    counter = 0
    min_val_loss = inf

    while counter < tolerance:
        train_one_epoch_tft(tft, tft_train_loader, tft_optimizer)
        val_loss = evaluate_one_epoch_tft(tft, tft_val_loader)
        if val_loss >= min_val_loss:
            counter += 1
        else:
            counter = 0
            min_val_loss = val_loss
            tft_copy = deepcopy(tft.state_dict())

        tft_scheduler.step()

    mlp.load_state_dict(mlp_copy)
    lstm.load_state_dict(lstm_copy)
    tft.load_state_dict(tft_copy)

    mlp_loss, mlp_preds = test_mlp(mlp, mlp_test_dataloader, scaling_factor, target_min)
    lstm_loss, lstm_preds = test_lstm(lstm, lstm_test_loader, scaling_factor, target_min)
    tft_loss, tft_preds = test_tft(tft, tft_test_loader, scaling_factor, target_min)

    bs = BS()
    bs.to(device)
    s, t, d = get_bs_data(test_from, test_to)
    s = tensor(s)
    t = tensor(t)
    bs_preds = bs(s)
    bs_loss = (t - bs_preds) ** 2

    rescale_factor = 1  # (mlp_target_scaler.data_max_ - mlp_target_scaler.data_min_) ** 2

    wandb.log(
        {
            'mlp_loss': mlp_loss * rescale_factor,
            'lstm_loss': lstm_loss * rescale_factor,
            'tft_loss': tft_loss * rescale_factor,
            'bs_loss': bs_loss.mean(),
        }
    )

    S, K, T, r, sigma = s.T

    mlp.to('cpu')
    lstm.to('cpu')
    tft.to('cpu')

    save(mlp.state_dict(), 'mlp.pt')
    save(lstm.state_dict(), 'lstm.pt')
    save(tft.state_dict(), 'tft.pt')

    mlp_artifact = wandb.Artifact(f'mlp_model_{test_from.strftime('%Y-%m-%d')}', type='model')
    mlp_artifact.add_file('mlp.pt')
    lstm_artifact = wandb.Artifact(f'lstm_model_{test_from.strftime('%Y-%m-%d')}', type='model')
    lstm_artifact.add_file('lstm.pt')
    tft_artifact = wandb.Artifact(f'tft_model_{test_from.strftime('%Y-%m-%d')}', type='model')
    tft_artifact.add_file('tft.pt')

    wandb.log_artifact(mlp_artifact)
    wandb.log_artifact(lstm_artifact)
    wandb.log_artifact(tft_artifact)

    data = stack(
        [
            S.to('cpu'),
            K.to('cpu'),
            T.to('cpu'),
            r.to('cpu'),
            sigma.to('cpu'),
            t.to('cpu'),
            mlp_preds.to('cpu'),
            lstm_preds.to('cpu'),
            tft_preds.to('cpu'),
            bs_preds.to('cpu'),
        ]
    ).T.detach()

    df = DataFrame(data=data.to('cpu'), columns=['S', 'K', 'T', 'r', '30RV', 'Target', 'MLP', 'LSTM', 'TFT', 'BS'])

    df['Date'] = d

    table = wandb.Table(dataframe=df)

    predictions = wandb.Artifact(
        f'Predictions_{test_from.strftime('%Y-%m-%d')}_{test_to.strftime('%Y-%m-%d')}', type='predictions'
    )
    predictions.add(table, 'predictions')
    wandb.log_artifact(predictions)


if __name__ == '__main__':
    main()
