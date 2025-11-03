import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_hourly_data
from aignosi_case.plots import save_current_figure

sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})
df_hourly = load_hourly_data()

target_variable = '% Silica Concentrate'
lags_to_plot = 48

# ACF/PACF require continuous time series without gaps
df_filled = df_hourly[target_variable].interpolate(method='linear')

# Forward/backward fill any remaining NaNs at edges
if df_filled.isnull().any():
    df_filled = df_filled.ffill().bfill()

# ACF: correlation of series with itself at different time lags
fig, ax = plt.subplots()
plot_acf(df_filled, lags=lags_to_plot, ax=ax, title=f'Função de Autocorrelação (ACF) - {target_variable}')
plt.xlabel('Lag (Horas)')
plt.ylabel('Autocorrelação')
plt.grid(True)
plt.tight_layout()
save_current_figure('autocorrelation_acf.png')
plt.show()

# PACF: partial correlation controlling for intermediate lags (Yule-Walker method)
fig, ax = plt.subplots()
plot_pacf(df_filled, lags=lags_to_plot, ax=ax, method='ywm', title=f'Função de Autocorrelação Parcial (PACF) - {target_variable}')
plt.xlabel('Lag (Horas)')
plt.ylabel('Autocorrelação Parcial')
plt.grid(True)
plt.tight_layout()
save_current_figure('autocorrelation_pacf.png')
plt.show()