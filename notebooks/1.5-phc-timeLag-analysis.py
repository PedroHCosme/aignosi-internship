import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_hourly_data, save_interim_data
from aignosi_case.plots import save_current_figure

# seaborn style settings
sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})

# load hourly resampled data using centralized utility
df_hourly = load_hourly_data()


# lags de 0 a 168 horas
MAX_LAG_HOURS = 168
output_var = '% Silica Concentrate'
input_vars = [
    '% Iron Feed',
    '% Silica Feed',        # A perturbação principal
    'Starch Flow',          # O controlador principal
    'Amina Flow',           # O segundo controlador
    'Flotation Column 01 Air Flow' # A variável de processo com maior corr. instantânea
]


lag_results = {}

# --- 2. Cálculo dos Lags ---

# Loop para cada variável de entrada que queremos testar
for input_var in input_vars:
    print(f"Calculando lags para: {input_var}")
    
    correlations = {}
    
    for lag in range(0, MAX_LAG_HOURS + 1):
        corr = df_hourly[output_var].corr(df_hourly[input_var].shift(lag))
        correlations[lag] = corr
        
    # Salva a série de correlações (Lag -> Valor) nos resultados
    lag_results[input_var] = pd.Series(correlations)

lag_df = pd.DataFrame(lag_results)


plt.figure(figsize=(14, 8))
lag_df.plot(
    grid=True,
    marker='o',
    markersize=4
)
plt.title(f'Análise de Correlação com Lag (Atraso) vs. "{output_var}"', fontsize=16)
plt.xlabel('Lag (Horas)')
plt.ylabel('Valor da Correlação')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5) # Linha zero
plt.legend(title='Variáveis de Entrada')

# Save lag correlation results
save_interim_data(lag_df, 'lag_correlation_results.csv')
save_current_figure('lag_correlation_analysis.png')
plt.show()


print("\n" + "="*50)
print("Resultados da Análise de Lag (Picos)")
print("="*50)

for column in lag_df.columns:
    # Encontra o lag com a maior correlação
    peak_abs_lag = lag_df[column].abs().idxmax()
    corr_at_peak = lag_df.loc[peak_abs_lag, column]
    print(f"Variável: {column}")
    print(f"  Pico de Correlação: {corr_at_peak:.4f}")
    print(f"  Encontrado em:      {peak_abs_lag} horas de lag\n")