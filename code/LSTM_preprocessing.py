from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from nelson_siegel_svensson.calibrate import calibrate_ns_ols
from pandas_market_calendars import get_calendar

holidays = list(get_calendar('CBOE_Index_Options').holidays().holidays)


def compute_time_idx(row: 'pd.Series[Any]', date_key: str, min_date: datetime) -> int:
    """
    Computes the number of bussiness days since min_date

        Args:
            row (Series): a row in a Dataframe
            date_key (str): expects name of column containg relevant date
            min_date (datetime): designates which date to offset from

        Returns:
            time_offset (int): bussiness days elapsed since min_date
    """
    return np.busday_count(min_date, row[date_key].date(), holidays=holidays)


def compute_time_idx_yields(row: 'pd.Series[Any]', date_key: str, min_date: datetime) -> int:
    """
    Computes the number of bussiness days since min_date

        Args:
            row (Series): a row in a Dataframe
            date_key (str): expects name of column containg relevant date
            min_date (datetime): designates which date to offset from

        Returns:
            time_offset (int): bussiness days elapsed since min_date
    """
    return np.busday_count(min_date, row[date_key].date())


def create_empty_rows(df: pd.DataFrame, idx: str, date_key: str | None = None) -> pd.DataFrame:
    """
    Reindexes on idx and creates rows filled with na values where there is a gap.
    If date_key is specified also fills new rows with business day dates in said column.

        Args:
            df (Dataframe): the df to create empty entries for
            idx (str): name of column to reindex on

        Optional args:
            date_key (str): name of column to fill in dates for

        Returns:
            df_copy (Dataframe): copy of df with empty rows where there were gaps in idx column
    """
    min_idx = df[idx].min()
    max_idx = df[idx].max()

    df = df.set_index(idx).reindex(range(min_idx, max_idx + 1))

    if date_key:
        min_date = df[date_key].min()
        max_date = df[date_key].max()
        df[date_key] = pd.bdate_range(start=min_date, end=max_date, freq='C', holidays=holidays)

    return df


def create_empty_rows_yields(df: pd.DataFrame, idx: str, date_key: str | None = None) -> pd.DataFrame:
    """
    Reindexes on idx and creates rows filled with na values where there is a gap.
    If date_key is specified also fills new rows with business day dates in said column.

        Args:
            df (Dataframe): the df to create empty entries for
            idx (str): name of column to reindex on

        Optional args:
            date_key (str): name of column to fill in dates for

        Returns:
            df_copy (Dataframe): copy of df with empty rows where there were gaps in idx column
    """
    min_idx = df[idx].min()
    max_idx = df[idx].max()

    df = df.set_index(idx).reindex(range(min_idx, max_idx + 1))

    if date_key:
        min_date = df[date_key].min()
        max_date = df[date_key].max()
        df[date_key] = pd.bdate_range(start=min_date, end=max_date)

    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with missing values interpolated.
    """
    return df.interpolate('linear')


if __name__ == '__main__':
    # Load dataset
    data = pd.DataFrame(pd.read_hdf('./code/data.h5', 'sorted_with_aggs'))
    print('Dataset:\n Loaded...')

    # Filter for 2011 as this is currently poc and we want fast runtimes
    data = data[(data['QUOTE_DATE'] < '2013-08-01')]
    min_date: datetime = data['QUOTE_DATE'].min().date()
    data['TIME_IDX'] = data.apply(compute_time_idx, args=('QUOTE_DATE', min_date), axis=1)

    data = data[(data['QUOTE_DATE'] != '2012-10-29') & (data['QUOTE_DATE'] != '2012-10-30')]
    underlying = data[['QUOTE_DATE', 'UNDERLYING_LAST', 'TIME_IDX']]

    data = data.dropna(subset=['C_LAST'])
    data = data[(data['Moneyness'] > 0.8) & (data['Moneyness'] < 2)]
    data = data[(data['TTM'] > 0) & (data['TTM'] < 2)]
    print(' Filtered...')

    # Load daily treassury yields
    daily_yields = pd.read_csv('./yield_data/daily-treasury-rates.csv')
    daily_yields = daily_yields.iloc[:, daily_yields.columns != 'Unnamed: 0']
    # Make Date column datetime
    daily_yields['Date'] = pd.to_datetime(daily_yields['Date'])
    # Compute time index
    min_date: datetime = daily_yields['Date'].min().date()
    daily_yields['TIME_IDX'] = daily_yields.apply(compute_time_idx_yields, args=('Date', min_date), axis=1)
    # Interpolate missing values
    daily_yields = create_empty_rows_yields(daily_yields, 'TIME_IDX', 'Date')
    daily_yields = fill_missing_values(daily_yields)
    print('Daily yields preprocessed')

    # Calculate risk free rate using Nelson Siegel for each datapoint
    maturities = np.array([1 / 12, 3 / 12, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    yield_curves = {}

    for index, row in daily_yields.iterrows():
        values = row[1:].to_numpy(dtype=np.float64)
        curve, status = calibrate_ns_ols(maturities, values)
        assert status.success
        yield_curves[row['Date'].strftime('%Y-%m-%d')] = curve

    data['RISK_FREE_RATE'] = [yield_curves[x['QUOTE_DATE'].strftime('%Y-%m-%d')](x['TTM']) for _, x in data.iterrows()]
    print('RFR added')

    # Save preprocessed lstm data
    data[data['QUOTE_DATE'] < '2013-01-01'].to_hdf('./code/lstm.h5', 'train_data')
    data[data['QUOTE_DATE'] >= '2013-01-01'].to_hdf('./code/lstm.h5', 'val_data')

    print('Data saved')

    # Get data on underlying for the time period, preprocess and save
    underlying = underlying.groupby('QUOTE_DATE').first().reset_index()
    min_date: datetime = underlying['QUOTE_DATE'].min().date()
    underlying = create_empty_rows(underlying, 'TIME_IDX', 'QUOTE_DATE')
    underlying = fill_missing_values(underlying)
    underlying['RETURNS'] = underlying['UNDERLYING_LAST'].diff()
    underlying[underlying['QUOTE_DATE'] < '2013-01-01'].to_hdf('./code/lstm.h5', 'train_underlying')
    underlying[underlying['QUOTE_DATE'] >= '2013-01-01'].to_hdf('./code/lstm.h5', 'val_underlying')
    print('Underlying saved')
