from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn style settings
sns.set_theme(style="whitegrid", rc={'figure.figsize':(12,6)})

# load dataset and setting first column as datetime and index
csv_filename = 'MiningProcess_Flotation_Plant_Database.csv'
workspace_root = Path('/home/pedrocosme/aignosi/aignosi-case')  
data_path = workspace_root / 'data' / 'raw' / csv_filename

df = pd.read_csv(data_path, parse_dates=[0], index_col=0, decimal=',')

print("Dataset Head")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nDataset Description")
print(df.describe())

# downsampling to hourly frequency by aggregating 20-second data using mean over each hour
df_hourly = df.resample('h').mean()
print("\nHourly Resampled Dataset Head")
print(df_hourly.head())

# checking for missing values after resampling (something could've gone wrong with the sensors)
print("\nMissing Values After Resampling")
print(df_hourly.isnull().sum())