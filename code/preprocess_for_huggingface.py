from datetime import date
from datetime import datetime
from typing import Any
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from nelson_siegel_svensson.calibrate import betas_ns_ols
from nelson_siegel_svensson.calibrate import errorfn_ns_ols
from pandas_market_calendars import get_calendar
from scipy.optimize import minimize

holidays = list(get_calendar('CBOE_Index_Options').holidays().holidays)


def _assert_same_shape(t: np.ndarray, y: np.ndarray) -> None:
    assert t.shape == y.shape, 'Mismatching shapes of time and values'


def calibrate_ns_ols(t: np.ndarray, y: np.ndarray, tau0: float = 2.0) -> Tuple[Any, Any]:
    """Calibrate a Nelson-Siegel curve to time-value pairs
    t and y, by optimizing tau and chosing all betas
    using ordinary least squares.
    """
    _assert_same_shape(t, y)
    opt_res = minimize(errorfn_ns_ols, x0=tau0, args=(t, y), bounds=((0.01, None),))
    curve, lstsq_res = betas_ns_ols(opt_res.x[0], t, y)
    return curve, opt_res


def compute_time_idx(df: pd.DataFrame, date_key: str, min_date: datetime, holidays: List) -> pd.Series:
    """
    Computes the number of bussiness days since min_date

        Args:
            df (DataFrame): a pandas Dataframe
            date_key (str): expects name of column containg relevant date
            min_date (datetime): designates which date to offset from
            holidays (List[Datetime]): A list of dates to be considered holidays
        Returns:
            time_offset (int): bussiness days elapsed since min_date
    """
    return np.busday_count(
        np.array([min_date for _ in range(len(df))], dtype=np.datetime64),
        df[date_key].values.astype('datetime64[D]'),
        holidays=holidays,
    )


def create_empty_rows(df: pd.DataFrame, idx: str, date_key: str | None = None, holidays: List = []) -> pd.DataFrame:
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


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with missing values interpolated.
    """
    return df.interpolate('linear')


data = pd.DataFrame(pd.read_hdf('./code/data.h5', 'sorted_with_aggs'))
# data = pd.DataFrame(pd.read_csv('hugging_v3.csv'))

print('loaded')

# These dates have no s&p500 trading data and should be excluded
non_trading_days = [
    '2012-10-29',
    '2012-10-30',
    '2018-12-05',
    '2022-01-17',
    '2022-02-21',
    '2022-04-15',
    '2022-05-30',
    '2022-06-20',
    '2022-07-04',
    '2022-09-05',
    '2022-11-24',
    '2022-12-26',
]
data = data[~data['QUOTE_DATE'].isin(non_trading_days)]

min_date: date = data['QUOTE_DATE'].min().date()
data['TIME_IDX'] = compute_time_idx(data, 'QUOTE_DATE', min_date, holidays)
print('Time idx added')
print('Empty dropped')

# Load daily treassury yields
daily_yields = pd.read_csv('./yield_data/daily-treasury-rates.csv')
# Make Date column datetime
daily_yields['Date'] = pd.to_datetime(daily_yields['Date'])
# Compute time index
min_date: date = daily_yields['Date'].min().date()
daily_yields['TIME_IDX'] = compute_time_idx(daily_yields, 'Date', min_date, [])
# Interpolate missing values
daily_yields = create_empty_rows(daily_yields, 'TIME_IDX', 'Date', [])
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

data['RISK_FREE_RATE'] = [
    yield_curves[x['QUOTE_DATE'].strftime('%Y-%m-%d')](x['TTM'])
    if x['TTM'] > 1 / 12
    else yield_curves[x['QUOTE_DATE'].strftime('%Y-%m-%d')](1 / 12) * 12 * x['TTM']
    for _, x in data.iterrows()
]
print('RFR added')


data.to_csv('hugging_v4.csv', index=False)
print('Done')

underlying = data[['QUOTE_DATE', 'UNDERLYING_LAST', 'TIME_IDX']]

underlying = underlying.groupby('QUOTE_DATE').first().reset_index()
duplicates = underlying[underlying['TIME_IDX'].duplicated()]['TIME_IDX'].values.tolist()

print(underlying[underlying['TIME_IDX'].isin(duplicates)])
underlying = create_empty_rows(underlying, 'TIME_IDX', 'QUOTE_DATE', holidays=holidays)
underlying = fill_missing_values(underlying)
underlying['RETURNS'] = underlying['UNDERLYING_LAST'].diff() / underlying['UNDERLYING_LAST'].shift(
    1, fill_value=1132.99
)
underlying.fillna(0, inplace=True)
underlying['30RV'] = underlying['RETURNS'].rolling(30).std()
underlying.to_csv('underlying.csv', index=False)
print('Underlying saved')

data = data[~data['C_LAST'].isna()]
data = data.merge(underlying[['QUOTE_DATE', '30RV']], on='QUOTE_DATE')

data.to_hdf('finalv2.h5', 'data')
underlying.to_hdf('finalv2.h5', 'underlying')
