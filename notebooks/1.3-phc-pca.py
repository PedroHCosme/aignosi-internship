import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_hourly_data, save_interim_data
from aignosi_case.plots import save_current_figure

# seaborn style settings
sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})

# load hourly resampled data using centralized utility
df_hourly = load_hourly_data()

#  Preparação dos Dados para PCA 

# removendo as 318 horas do shutdown
df_pca_ready = df_hourly.dropna()

# definindo y e X
y_target_series = df_pca_ready['% Silica Concentrate']
X_features = df_pca_ready.drop(columns=['% Iron Concentrate', '% Silica Concentrate'])

# Scaling para o PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

print(f"\nRodando PCA em {X_scaled.shape[1]} features (variáveis)...")

# Análise de Variância Explicada

# PCA com todos os componentes
pca_full = PCA()
pca_full.fit(X_scaled)

# variância acumulada
explained_variance_cumulative = np.cumsum(pca_full.explained_variance_ratio_)

# Encontrar quantos componentes são necessários para 95% da variância
n_components_95 = np.where(explained_variance_cumulative >= 0.95)[0][0] + 1
print(f"\n--- PCA ---")
print(f"São necessários {n_components_95} componentes para explicar 95% da variância")
print(f"Conclusão: Conseguimos agrupar {X_scaled.shape[1]} variáveis em {n_components_95}")

# Plotar a Curva de Variância
plt.figure(figsize=(12, 7))
plt.plot(explained_variance_cumulative, marker='o', linestyle='--')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Explicada Cumulativa')
plt.title('Variância Explicada vs. Número de Componentes')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variância')
plt.axvline(x=n_components_95 - 1, color='g', linestyle='--', label=f'{n_components_95} Componentes')
plt.legend()
save_current_figure('pca_variance.png')
plt.show()

# Visualização 2D

# PCA com 2 componentes
pca_2 = PCA(n_components=2)
X_pca_2d = pca_2.fit_transform(X_scaled)

df_pca_plot = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
df_pca_plot['Silica (Alvo)'] = y_target_series.values 

# variância dos 2 primeiros componentes
pc1_var = pca_2.explained_variance_ratio_[0] * 100
pc2_var = pca_2.explained_variance_ratio_[1] * 100
print(f"PC1 explica {pc1_var:.2f}% da variância.")
print(f"PC2 explica {pc2_var:.2f}% da variância.")
print(f"Total (PC1+PC2): {pc1_var + pc2_var:.2f}% da variância total.")

plt.figure(figsize=(14, 9))
sns.scatterplot(
    data=df_pca_plot,
    x='PC1',
    y='PC2',
    hue='Silica (Alvo)',
    palette='coolwarm', 
    alpha=0.6,
    s=20 
)
plt.title(f'PCA (PC1 vs PC2) Colorido pela Concentração de Sílica (Alvo)\nPC1+PC2 explicam {pc1_var + pc2_var:.2f}% da variância', fontsize=16)
plt.xlabel(f'Componente Principal 1 ({pc1_var:.2f}%)')
plt.ylabel(f'Componente Principal 2 ({pc2_var:.2f}%)')
plt.legend(title='Sílica (%)')
plt.grid(True)
save_current_figure('pca_2d_scatter.png')
plt.show()

save_interim_data(df_pca_plot, 'pca_2d_data.csv')