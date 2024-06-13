from datetime import date

import torch
import torch.utils
import torch.utils.data
from datasets import MLPDataset
from utils import get_mlp_data

import wandb


def main():
    wandb.init(project='cross_validate')
    data, targets = get_mlp_data(date(2011, 1, 1), date(2011, 6, 1))

    u_max = targets.max()
    u_min = targets.min()
    targets = (targets - u_min) / (u_max - u_min)

    dataset = MLPDataset(data, targets, 'mps')
    dataloader = torch.utils.data.DataLoader(dataset, 1)

    tables = wandb.Table(columns=['Target', 'a', 'b'])

    preds = wandb.Artifact('preds', 'predictions')

    for _, (_, t) in enumerate(dataloader):
        a, b = torch.randn((1,)), torch.randn((1,))
        tables.add_data(t.item(), a.item(), b.item())

    preds.add(tables, 'table')
    wandb.log_artifact(preds)


if __name__ == '__main__':
    main()
