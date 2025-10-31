import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_hourly_data
from aignosi_case.plots import save_current_figure

# seaborn style settings
sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})

# load hourly resampled data using centralized utility
df_hourly = load_hourly_data()


# Vamos focar no nosso alvo, na entrada principal, e em um controlador-chave
# Usar o df_hourly ORIGINAL (com NaNs) para ver o shutdown
key_cols = [
        '% Silica Concentrate', # Alvo
        '% Silica Feed',        # Principal Entrada de Impureza
        'Starch Flow',          # Um dos principais Reagentes 
        'Amina Flow'            # Outro Reagente
    ]


df_hourly[key_cols].plot(
    subplots=True, 
    layout=(-1, 1),  
    figsize=(16, 12),
    title='Análise de Série Temporal (Março a Setembro 2017)',
    grid=True
)
plt.xlabel('Data')
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
save_current_figure('time_series_key_variables.png')
plt.show()


# Outliers e Distribuição (Boxplots)
plt.figure(figsize=(14, 8))
df_melted = df_hourly[key_cols].melt(var_name='Variável', value_name='Valor')

sns.boxplot(data=df_melted, x='Variável', y='Valor')
plt.title('Análise de Distribuição e Outliers (Horária)', fontsize=16)
plt.ylabel('Valores das Unidades')
plt.xlabel('')
plt.xticks(rotation=10)
save_current_figure('boxplot_outliers_distribution.png')
plt.show()

# Análise de Sazonalidade (Zoom em 1 Semana)
df_weekly_zoom = df_hourly.loc['2017-07-01':'2017-07-07']

df_weekly_zoom[key_cols].plot(
    subplots=True,
    layout=(-1, 1),
    figsize=(16, 12),
    title='Zoom de 1 Semana (Análise de Sazonalidade Diária)',
    grid=True,
    marker='o', 
    markersize=3
)
plt.xlabel('Data e Hora')
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
save_current_figure('weekly_zoom_seasonality.png')
plt.show()