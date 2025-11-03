import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_hourly_data, save_interim_data
from aignosi_case.plots import save_current_figure

sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})
df_hourly = load_hourly_data()

# Test up to 1 week lag to capture delayed process dynamics
MAX_LAG_HOURS = 168
output_var = '% Silica Concentrate'
input_vars = [
    '% Iron Feed',
    '% Silica Feed',
    'Starch Flow',
    'Amina Flow',
    'Flotation Column 01 Air Flow'
]

lag_results = {}

for input_var in input_vars:
    print(f"Calculando lags para: {input_var}")
    
    correlations = {}
    
    # Compute Pearson correlation between output(t) and input(t-lag)
    for lag in range(0, MAX_LAG_HOURS + 1):
        corr = df_hourly[output_var].corr(df_hourly[input_var].shift(lag))
        correlations[lag] = corr
    
    lag_results[input_var] = pd.Series(correlations)

# Transform dict of Series into DataFrame (rows=lags, cols=variables)
lag_df = pd.DataFrame(lag_results)


plt.figure(figsize=(14, 8))
lag_df.plot(grid=True, marker='o', markersize=4)
plt.title(f'Análise de Correlação com Lag (Atraso) vs. "{output_var}"', fontsize=16)
plt.xlabel('Lag (Horas)')
plt.ylabel('Valor da Correlação')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.legend(title='Variáveis de Entrada')

save_interim_data(lag_df, 'lag_correlation_results.csv')
save_current_figure('lag_correlation_analysis.png')
plt.show()

print("\n" + "="*50)
print("Resultados da Análise de Lag (Picos)")
print("="*50)

for column in lag_df.columns:
    # Find lag with maximum absolute correlation (strongest relationship)
    peak_abs_lag = lag_df[column].abs().idxmax()
    corr_at_peak = lag_df.loc[peak_abs_lag, column]
    print(f"Variável: {column}")
    print(f"  Pico de Correlação: {corr_at_peak:.4f}")
    print(f"  Encontrado em:      {peak_abs_lag} horas de lag\n")