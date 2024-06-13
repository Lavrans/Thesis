from numpy import array
from numpy import float32 as np_float32
from pandas import DataFrame
from pandas import read_hdf
from torch import Tensor
from torch import float32
from torch import tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, data: DataFrame, n_timesteps: int) -> None:
        self.n_timesteps = n_timesteps
        self.data = self.preprocess_data(data)

    def preprocess_data(self, data: DataFrame) -> list[DataFrame]:
        # Sort data by 'QUOTE_UNIXTIME'
        print('preprocessing....')
        data = data.sort_values(by='QUOTE_UNIXTIME')
        print('sorted....')
        # Group data by 'EXPIRE_DATE' and 'STRIKE'
        grouped_data = data.groupby(['EXPIRE_DATE', 'STRIKE'])
        print('grouped....')
        sequences = []

        for _, group_data in grouped_data:
            # Create sequences of n timesteps for each group
            for i in range(len(group_data) - self.n_timesteps):
                sequence = group_data.iloc[i : i + self.n_timesteps + 1]  # Include the next timestep
                sequences.append(sequence)
        print('sequenced....')
        return sequences

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sequence = self.data[idx]

        # Extract features and target variable (e.g., 'UNDERLYING_LAST')
        features = sequence[
            [
                'C_DELTA',
                'C_GAMMA',
                'C_VEGA',
                'C_THETA',
                'C_RHO',
                'C_IV',
                'C_VOLUME',
                'C_LAST',
                'C_BID',
                'C_ASK',
                'P_DELTA',
                'P_GAMMA',
                'P_VEGA',
                'P_THETA',
                'P_RHO',
                'P_IV',
                'P_VOLUME',
                'P_LAST',
                'P_BID',
                'P_ASK',
            ]
        ]

        features = array(features.values, dtype=np_float32)

        target = array(sequence.iloc[-1]['STRIKE'], dtype=np_float32)

        return tensor(features, dtype=float32), tensor(target, dtype=float32)


df = DataFrame(read_hdf('data.h5', 'cleaned_with_aggs'))

print(len(df))

print(df.dtypes)

df = df.sort_values(by='QUOTE_UNIXTIME').head(5000)
n_timesteps = 5
dataset = TimeSeriesDataset(df, n_timesteps)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

dataiter = iter(dataloader)

features, target = next(dataiter)

print(f'Features: {features}\n Target: {target}')
