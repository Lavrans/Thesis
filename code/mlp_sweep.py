from datetime import date
from math import isnan

from datasets import MLPDataset
from models import MLP
from numpy import inf
from sklearn.preprocessing import MinMaxScaler
from torch import cuda
from torch import device as torch_device
from torch import nn
from torch import no_grad
from torch.backends import mps
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from utils import get_mlp_data

import wandb

# Define sweep config
# sweep_configuration = {
#     'program': 'code/mlp_sweep.py',
#     'method': 'bayes',
#     'name': 'sweep',
#     'metric': {'goal': 'minimize', 'name': 'val_loss.min'},
#     'parameters': {
#         'batch_size': {'values': [1024, 2048, 4096]},
#         'lr': {'max': 0.1, 'min': 0.0001},
#         'gamma': {'max': 1.0, 'min': 0.85},
#         'num_layers': {'max': 5, 'min': 1},
#         'hidden_size': {'values': [10, 50, 100, 200]},
#         'leaky_alpha': {'max': 0.5, 'min': 0.0},
#     },
# }

# # Initialize sweep by passing in config.
# # (Optional) Provide a name of the project.
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='mlp-sweep')

criterion = nn.MSELoss()
device = torch_device('cuda' if cuda.is_available() else ('mps' if mps.is_available() else 'cpu'))
print(f'{device}')

train_data, train_targets = get_mlp_data(date(2012, 1, 1), date(2014, 10, 1))
val_data, val_targets = get_mlp_data(date(2014, 10, 1), date(2015, 1, 1))

mlp_scaler = MinMaxScaler()
train_data = mlp_scaler.fit_transform(train_data)
val_data = mlp_scaler.transform(val_data)

# target_scaler = MinMaxScaler()
# train_data[['C_PRICE']] = target_scaler.fit_transform(train_data[['C_PRICE']])
# val_data[['C_PRICE']] = target_scaler.transform(val_data[['C_PRICE']])

train_dataset = MLPDataset(train_data, train_targets, device)
val_dataset = MLPDataset(val_data, val_targets, device)


# Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a
# model and return metric
def train_one_epoch(model, train_loader, optimizer):
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


def evaluate_one_epoch(model, val_loader):
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


def main():
    wandb.init()
    wandb.define_metric('val_loss', summary='min,max,mean,last')

    # note that we define values from `wandb.config`
    # instead of defining hard values
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    gamma = wandb.config.gamma
    num_layers = wandb.config.num_layers
    hidden_size = wandb.config.hidden_size
    leaky_alpha = wandb.config.leaky_alpha
    model = MLP(num_layers, hidden_size, leaky_alpha)
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


if __name__ == '__main__':
    main()

# wandb.agent(sweep_id, function=main, count=150)
