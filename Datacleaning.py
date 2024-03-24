import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

input_dir = './data'
output_dir = './redata'

input_files = ['Average_House_Prices.csv', 'Inflation.csv', 'InteresRate.csv', 'MORTGAGE.csv', 'Supply_NewHouse.csv','UnemploymentRATE.csv', 'Population.csv']

output_files = ['Average_House_Prices1.csv', 'Inflation1.csv', 'InteresRate1.csv', 'MORTGAGE1.csv', 'Supply_NewHouse1.csv', 'UnemploymentRATE1.csv', 'Population1.csv']

for i in range(len(input_files)):
    input_path = os.path.join(input_dir, input_files[i])
    output_path = os.path.join(output_dir, output_files[i])

    data = pd.read_csv(input_path)
    data['DATE'] = pd.to_datetime(data['DATE'])

    years = data['DATE'].dt.year
    months = data['DATE'].dt.month

    avg_data_multi_idx = data.groupby([years, months], as_index=True).mean(numeric_only=True)
    avg_data = data.groupby([years, months], as_index=False).mean(numeric_only=True)
    multi_idx = list(avg_data_multi_idx.index)

    avg_years = [idx[0] for idx in multi_idx]
    avg_months = [idx[1] for idx in multi_idx]

    avg_data['YEAR'] = avg_years
    avg_data['MONTH'] = avg_months

    start_Year = 1990.0
    end_Year = 2023.0

    avg_data_filtered = avg_data[(avg_data['YEAR'] >= start_Year) & (avg_data['YEAR'] <= end_Year)]
    avg_data_filtered['DAY']= 1
    avg_data_filtered['date'] = pd.to_datetime(avg_data_filtered[['YEAR', 'MONTH','DAY']])

    avg_data_filtered = avg_data_filtered.drop(['YEAR', 'MONTH', 'DAY'], axis=1)

    avg_data_filtered.to_csv(output_path, index=False)

df = pd.merge(pd.read_csv(os.path.join(output_dir, output_files[0])),
              pd.read_csv(os.path.join(output_dir, output_files[1])),
              on='date', how='left')

for f in output_files[2:]:
    df = pd.merge(df, pd.read_csv(os.path.join(output_dir, f)),
                  on='date', how='left')

df.set_index('date', inplace=True)
df.to_csv('./redata/House.csv', index=True)


