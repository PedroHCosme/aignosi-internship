import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_hourly_data
from aignosi_case.plots import save_current_figure

sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})
df_hourly = load_hourly_data()

df_analysis = df_hourly.dropna().copy()

control_vars = [
    'Starch Flow', 
    'Amina Flow',
    'Ore Pulp pH',
    'Ore Pulp Density',
    'Flotation Column 01 Air Flow',
    'Flotation Column 01 Level',
    'Flotation Column 07 Air Flow',
    'Flotation Column 07 Level'
]

# Transform wide format to long format for seaborn violin plot
df_melted_control = df_analysis[control_vars].melt(var_name='Variável', value_name='Valor')

plt.figure()
sns.violinplot(
    data=df_melted_control, 
    x='Variável', 
    y='Valor', 
    inner='quartile',  # Display median and quartiles inside violin
    scale='width',     # Normalize violin widths for comparison
    palette='Set3'
)
plt.title('Distribuição das Variáveis de Controle e Processo (Violin Plots)', fontsize=20)
plt.ylabel('Valores das Unidades (Escalas Diferentes)')
plt.xlabel('Variável de Processo')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y')
plt.tight_layout()
save_current_figure('violin_plot_control_variables.png')
plt.show()
