from datetime import date

import numpy as np
import pandas as pd
from numba import jit
from numpy.typing import NDArray


@jit
def get_arrays(data, underlying, timesteps):
    r = []
    s = []
    t = []
    for x in data:
        i = int(x[-1]) + 1
        r.append(underlying[i - timesteps : i])
        s.append(x[1:-1])
        t.append(x[0])
    return (r, s, t)


def get_lstm_data(min_date: date, max_date: date, timesteps: int = 140) -> tuple[NDArray, NDArray, NDArray]:
    data = pd.read_hdf('final_data.h5', 'data')
    underlying = pd.read_hdf('final_data.h5', 'underlying')

    data = data[~data['30RV'].isna()]
    data = data[~data['C_PRICE'].isna()]

    data = data[(data['QUOTE_DATE'].dt.date >= min_date) & (data['QUOTE_DATE'].dt.date < max_date)]
    data = data[(data['TTM'] > 0) & (data['TTM'] < 2)]
    data = data[(data['Moneyness'] > 0.8) & (data['Moneyness'] < 2)]

    data = data[['C_PRICE', 'UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', 'TIME_IDX']].to_numpy()
    underlying = underlying['RETURNS'].to_numpy()

    r, s, t = get_arrays(data, underlying, timesteps)
    r = np.array(r)
    s = np.array(s)
    t = np.array(t)
    return (r, s, t)


def get_tft_data(min_date: date, max_date: date, timesteps: int = 140) -> tuple[NDArray, NDArray, NDArray]:
    data = pd.read_hdf('final_data.h5', 'data')
    underlying = pd.read_hdf('final_data.h5', 'underlying')

    data = data[~data['30RV'].isna()]
    data = data[~data['C_PRICE'].isna()]

    data = data[(data['QUOTE_DATE'].dt.date >= min_date) & (data['QUOTE_DATE'].dt.date < max_date)]
    data = data[(data['TTM'] > 0) & (data['TTM'] < 2)]
    data = data[(data['Moneyness'] > 0.8) & (data['Moneyness'] < 2)]

    data = data[['C_PRICE', 'UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', '30RV', 'TIME_IDX']].to_numpy()
    underlying = underlying['RETURNS'].to_numpy()

    r, s, t = get_arrays(data, underlying, timesteps)
    r = np.array(r)
    s = np.array(s)
    t = np.array(t)
    return (r, s, t)


def get_mlp_data(min_date: date, max_date: date) -> tuple[NDArray, NDArray]:
    data = pd.read_hdf('final_data.h5', 'data')

    # We rely on IV data
    data = data[~data['30RV'].isna()]
    data = data[~data['C_PRICE'].isna()]

    data = data[(data['QUOTE_DATE'].dt.date >= min_date) & (data['QUOTE_DATE'].dt.date < max_date)]
    data = data[(data['TTM'] > 0) & (data['TTM'] < 2)]
    data = data[(data['Moneyness'] > 0.8) & (data['Moneyness'] < 2)]

    return data[['UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', '30RV']].to_numpy(), data['C_PRICE'].to_numpy()


def get_bs_data(min_date: date, max_date: date) -> tuple[NDArray, NDArray]:
    data = pd.read_hdf('final_data.h5', 'data')

    # We rely on IV data
    data = data[~data['30RV'].isna()]
    data = data[~data['C_PRICE'].isna()]

    data = data[(data['QUOTE_DATE'].dt.date >= min_date) & (data['QUOTE_DATE'].dt.date < max_date)]
    data = data[(data['TTM'] > 0) & (data['TTM'] < 2)]
    data = data[(data['Moneyness'] > 0.8) & (data['Moneyness'] < 2)]

    return (
        data[['UNDERLYING_LAST', 'STRIKE', 'TTM', 'RISK_FREE_RATE', '30RV']].to_numpy(),
        data['C_PRICE'].to_numpy(),
        data['QUOTE_DATE'].to_numpy(),
    )


if __name__ == '__main__':
    data = get_tft_data(date(2012, 1, 1), date(2015, 1, 1))
    print(data)
