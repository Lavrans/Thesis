from copy import deepcopy
from datetime import date
from math import sqrt

from datasets import MLPDataset
from models import MLP
from numpy import inf
from sklearn.preprocessing import MinMaxScaler
from torch import cuda
from torch import device as torch_device
from torch import nn
from torch import no_grad
from torch import save
from torch import tensor
from torch.backends import mps
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from utils import get_mlp_data

model = MLP(4, 100, 0.05)

device = torch_device('cuda' if cuda.is_available() else ('mps' if mps.is_available() else 'cpu'))
print(f'{device}')

model.to(device=device)

optimizer = Adam(model.parameters())
scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
criterion = nn.MSELoss()

train_data = get_mlp_data(date(2012, 1, 1), date(2015, 1, 1))

val_data = get_mlp_data(date(2015, 1, 1), date(2015, 2, 1))

test_data = get_mlp_data(date(2015, 2, 1), date(2015, 3, 1))

mlp_scaler = MinMaxScaler()
train_data[['UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', 'C_IV']] = mlp_scaler.fit_transform(
    train_data[['UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', 'C_IV']]
)
val_data[['UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', 'C_IV']] = mlp_scaler.transform(
    val_data[['UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', 'C_IV']]
)
test_data[['UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', 'C_IV']] = mlp_scaler.transform(
    test_data[['UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', 'C_IV']]
)

target_scaler = MinMaxScaler()
train_data[['C_PRICE']] = target_scaler.fit_transform(train_data[['C_PRICE']])
val_data[['C_PRICE']] = target_scaler.transform(val_data[['C_PRICE']])


print(f'data min: {mlp_scaler.data_min_}')
print(f'data max: {mlp_scaler.data_max_}')
print(f'target min: {target_scaler.data_min_}')
print(f'target max: {target_scaler.data_max_}')

train_dataset = MLPDataset(train_data)
val_dataset = MLPDataset(val_data)
test_dataset = MLPDataset(test_data)

train_loader = DataLoader(train_dataset, 4096, True)
val_loader = DataLoader(val_dataset, 4096, False)
test_loader = DataLoader(test_dataset, 4096, False)

train_losses = []
val_losses = []
min_val_loss = inf

tolerance = 20
counter = 0
epoch = 1

print(f"{'':-^64}")
print(f"|{'Epoch':^10}|{'Training loss':^25}|{'Validation loss':^25}|")
print(f"{'':-^64}")

while counter < tolerance:
    train_loss = 0
    val_loss = 0

    model.train()

    for idx, (s, t) in enumerate(train_loader):
        optimizer.zero_grad()
        x = s.float().to(device)
        pred = model(x)
        loss = criterion(pred.squeeze(), t.float().to(device))

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    mean_loss_train = train_loss / (idx + 1)
    train_losses.append(mean_loss_train)

    model.eval()

    with no_grad():
        for idx, (s, t) in enumerate(val_loader):
            optimizer.zero_grad()
            x = s.float().to(device)
            pred = model(x)
            loss = criterion(pred.squeeze(), t.float().to(device))
            val_loss += loss.item()
    mean_loss_val = val_loss / (idx + 1)
    val_losses.append(mean_loss_val)

    if mean_loss_val >= min_val_loss:
        counter += 1
        new_best = False
    else:
        counter = 0
        min_val_loss = mean_loss_val
        model.to('cpu')
        state_dict_copy = deepcopy(model.state_dict())
        save(state_dict_copy, 'mlp.pt')
        model.to(device)
        new_best = True

    print(f"|{epoch:^10d}|{mean_loss_train:>20.5e}{'':>5}|{mean_loss_val:>20.5e}{'':>5}|{'x' if new_best else ''}")

    scheduler.step()
    epoch += 1


# Run test
test_loss = 0

model.eval()

with no_grad():
    for idx, (s, t) in enumerate(test_loader):
        optimizer.zero_grad()
        x = s.float().to(device)
        pred = model(x).to('cpu')
        rescaled = tensor(target_scaler.inverse_transform(pred))
        loss = criterion(rescaled.squeeze(), t.float())
        test_loss += loss.item()
mean_loss_test = test_loss / (idx + 1)

model.to('cpu')
save(model.state_dict(), 'mlp_final.pt')

test_loss = 0

model.load_state_dict(state_dict_copy)
model.to(device)
model.eval()

with no_grad():
    for idx, (s, t) in enumerate(test_loader):
        optimizer.zero_grad()
        x = s.float().to(device)
        pred = model(x).to('cpu')
        print(pred.sum())
        rescaled = tensor(target_scaler.inverse_transform(pred))
        loss = criterion(rescaled.squeeze(), t.float())
        test_loss += loss.item()
mean_loss_test_best = test_loss / (idx + 1)

# Print and save final results

print(f"{'':-^64}")


print(f'Final MSE: {val_losses[-1]}\nFinal RMSE: {sqrt(val_losses[-1])}')
print(f'Best MSE: {min_val_loss}\nBest RMSE: {sqrt(min_val_loss)}')
print(f'Test MSE: {mean_loss_test}\nTest RMSE: {sqrt(mean_loss_test)}')
print(f'Best test MSE: {mean_loss_test_best}\nBest test RMSE: {sqrt(mean_loss_test_best)}')
