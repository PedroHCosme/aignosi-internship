import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_interim_data 
from aignosi_case.plots import save_current_figure

sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': (10, 7)})

# Load cluster means computed in 1.6-phc-Kmeans.py
cluster_analysis = load_interim_data('kmeans_cluster_analysis.csv')

# Move cluster index to column for plotting
plot_data = cluster_analysis.reset_index()

# Map cluster numbers to regime quality labels
regime_map = {
    0: 'Regime 0 (Padrão)',
    1: 'Regime 1 (Ótimo)',
    2: 'Regime 2 (Indesejável)'
}
plot_data['Regime'] = plot_data['Cluster'].map(regime_map)

# Order bars by quality: best to worst
regime_order = ['Regime 1 (Ótimo)', 'Regime 0 (Padrão)', 'Regime 2 (Indesejável)']
regime_palette = {
    'Regime 1 (Ótimo)': '#2ca02c',     # Green
    'Regime 0 (Padrão)': '#8c8c8c',    # Gray
    'Regime 2 (Indesejável)': '#d62728' # Red
}

plt.figure()
ax = sns.barplot(data=plot_data, x='Regime', y='% Silica Concentrate', order=regime_order, 
                 palette=regime_palette, hue='Regime', dodge=False, legend=False)

# Annotate each bar with its exact value
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points',
                fontweight='bold', fontsize=12)

ax.set_title('Resultado de Qualidade por Regime Operacional', fontsize=18, pad=20)
ax.set_xlabel('Regime Operacional', fontsize=12)
ax.set_ylabel('Média de % de Sílica (Impureza)', fontsize=12)
ax.set_ylim(0, plot_data['% Silica Concentrate'].max() * 1.25)

save_current_figure('regime_quality_barplot.png')
plt.tight_layout()
plt.show()