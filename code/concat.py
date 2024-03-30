import os

import pandas as pd

files = os.listdir('./yield_data')

for f in files:
    if '.csv' not in f:
        files.remove(f)

dfs = []

for f in files:
    dfs.append(
        pd.read_csv(os.path.join('yield_data', f))[
            ['Date', '1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
        ]
    )

df = pd.concat(dfs)


df['Date'] = pd.to_datetime(df['Date'])

df.sort_values(by=['Date'], inplace=True)

df.to_csv('./yield_data/daily-treasury-rates.csv', index=False)
