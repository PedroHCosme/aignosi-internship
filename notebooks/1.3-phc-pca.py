import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_hourly_data, save_interim_data
from aignosi_case.plots import save_current_figure

sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})
df_hourly = load_hourly_data()

df_pca_ready = df_hourly.dropna()
y_target_series = df_pca_ready['% Silica Concentrate']
X_features = df_pca_ready.drop(columns=['% Iron Concentrate', '% Silica Concentrate'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

print(f"\nRodando PCA em {X_scaled.shape[1]} features")

pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance_cumulative = np.cumsum(pca_full.explained_variance_ratio_)

n_components_95 = np.where(explained_variance_cumulative >= 0.95)[0][0] + 1
print(f"\n--- PCA ---")
print(f"São necessários {n_components_95} componentes para explicar 95% da variância")

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

# Extract PC1 loadings (eigenvector weights for first principal component)
loadings_pc1 = pca_full.components_[0]

loadings_df = pd.DataFrame({
    'Feature': X_features.columns,
    'Loading_PC1': loadings_pc1,
    'Abs_Loading': np.abs(loadings_pc1)
})

top_10_features = loadings_df.sort_values(by='Abs_Loading', ascending=False).head(10)
print("\nTop 10 variáveis mais influentes (PC1):")
print(top_10_features[['Feature', 'Loading_PC1']])

top_10_plot = top_10_features.sort_values(by='Loading_PC1', ascending=False)

plt.figure(figsize=(12, 8))
ax = sns.barplot(data=top_10_plot, x='Loading_PC1', y='Feature', palette='vlag', orient='h')
ax.set_title(f'Top 10 Variáveis que mais Influenciam o PC1 ({pca_full.explained_variance_ratio_[0]*100:.2f}% da Variância)', fontsize=16)
ax.set_xlabel('Peso (Loading) - Influência no PC1')
ax.set_ylabel('Variável de Processo')
plt.grid(True)
save_current_figure('pca_pc1_loadings_top10.png')
plt.show()

# Extract PC2 loadings (second eigenvector)
pc2_var_explained = pca_full.explained_variance_ratio_[1] * 100
loadings_pc2 = pca_full.components_[1]

loadings_df_pc2 = pd.DataFrame({
    'Feature': X_features.columns,
    'Loading_PC2': loadings_pc2,
    'Abs_Loading_PC2': np.abs(loadings_pc2)
})

top_10_pc2 = loadings_df_pc2.sort_values(by='Abs_Loading_PC2', ascending=False).head(10)
top_10_pc2_plot = top_10_pc2.sort_values(by='Loading_PC2', ascending=False)

print("\nTop 10 variáveis (PC2):")
print(top_10_pc2[['Feature', 'Loading_PC2']])

plt.figure(figsize=(12, 8))
sns.barplot(data=top_10_pc2_plot, x='Loading_PC2', y='Feature', palette='vlag', orient='h')
plt.title(f'Top 10 Variáveis que mais Influenciam o PC2 (Eixo Y) ({pc2_var_explained:.2f}%)', fontsize=16)
plt.xlabel('Peso (Loading) - Influência no PC2')
plt.ylabel('Variável de Processo')
plt.grid(True)
save_current_figure('pca_pc2_loadings_top10.png')
plt.show()

# Project data onto first 2 principal components for visualization
pca_2 = PCA(n_components=2)
X_pca_2d = pca_2.fit_transform(X_scaled)

df_pca_plot = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
df_pca_plot['Silica (Alvo)'] = y_target_series.values 

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