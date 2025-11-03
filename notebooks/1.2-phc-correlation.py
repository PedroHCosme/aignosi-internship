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

#  Matriz de Correlação
plt.figure(figsize=(18, 9))
corr_matrix = df_hourly.corr()

save_interim_data(corr_matrix, 'correlation_matrix.csv')

# máscara para a diagonal superior
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

#  heatmap
sns.heatmap(corr_matrix, 
            mask=mask,           
            annot=True,          
            fmt='.2f',           
            cmap='coolwarm',     
            vmin=-1, vmax=1,
            annot_kws={"size": 9}  )
plt.title('Matriz de Correlação', fontsize=12)  
plt.xticks(rotation=45, ha='right', fontsize=8)  
plt.yticks(rotation=0, fontsize=8)  
plt.tight_layout() 

save_current_figure('correlation_matrix.png')
plt.show()

# Análise de Correlação
print("\n" + "*"*50)
print("Correlações com target: '% Silica Concentrate'")
print("*"*50)

target_corr = corr_matrix['% Silica Concentrate'].drop('% Silica Concentrate')
print(target_corr.sort_values(ascending=False))

print("\n" + "*"*50)
print("Pares de Alta Correlação (Multicolinearidade)")
print("*"*50)

masked_corr = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))

stacked_corr = masked_corr.stack()

high_corr_pairs = stacked_corr[stacked_corr.abs() > 0.8]

if high_corr_pairs.empty:
    print("Nenhum par de variável com correlação > 0.8 encontrado.")
else:
    print(high_corr_pairs.sort_values(ascending=False))