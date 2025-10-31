import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_raw_data, load_hourly_data

# seaborn style settings
sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})

# load dataset using centralized utility
df = load_raw_data()

print("Dataset Head")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nDataset Description")
print(df.describe())

# load hourly resampled data using centralized utility
df_hourly = load_hourly_data()
print("\nHourly Resampled Dataset Head")
print(df_hourly.head())

# checking for missing values after resampling (something could've gone wrong with the sensors)
print("\nMissing Values After Resampling")
print(df_hourly.isnull().sum())