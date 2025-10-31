import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE, INTERIM_DATA_DIR
from aignosi_case.dataset import load_hourly_data, save_interim_data
from aignosi_case.plots import save_current_figure

sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})

df_hourly = load_hourly_data()
df_pca_plot = pd.read_csv(INTERIM_DATA_DIR / 'pca_2d_data.csv', index_col=0)

df_pca_ready = df_hourly.dropna()
X_features = df_pca_ready.drop(columns=['% Iron Concentrate', '% Silica Concentrate'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

IDEAL_K = 3

inertia_values = []
k_range = range(1, 11)

for k in k_range:
    kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans_model.fit(X_scaled)
    inertia_values.append(kmeans_model.inertia_)

plt.figure(figsize=(12, 7))
plt.plot(k_range, inertia_values, marker='o', linestyle='--')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Inércia (WCSS)')
plt.title('Método do Cotovelo para Encontrar o K Ideal')
plt.grid(True)
plt.xticks(k_range)
save_current_figure('kmeans_elbow.png')
plt.show()

kmeans_final = KMeans(n_clusters=IDEAL_K, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)

df_cluster_plot = df_pca_plot.copy()
df_cluster_plot['Cluster'] = cluster_labels
df_cluster_plot['Cluster'] = df_cluster_plot['Cluster'].astype('category')

plt.figure(figsize=(14, 9))
sns.scatterplot(data=df_cluster_plot, x='PC1', y='PC2', hue='Cluster', palette='Set1', alpha=0.7, s=20)
plt.title(f'Visualização PCA (PC1 vs PC2) Colorida por {IDEAL_K} Clusters', fontsize=16)
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster')
plt.grid(True)
save_current_figure('cluster_pca_visualization.png')
plt.show()

df_analysis = df_hourly.dropna().copy()
df_analysis['Cluster'] = cluster_labels
df_analysis['Cluster'] = df_analysis['Cluster'].astype('category')

cluster_analysis = df_analysis.groupby('Cluster').mean(numeric_only=True)

print("\n" + "="*50)
print(f"Análise dos {IDEAL_K} Regimes Operacionais")
print("="*50)
print(cluster_analysis)

cluster_size = df_analysis['Cluster'].value_counts(normalize=True).sort_index()
print("\nDistribuição dos Clusters:")
print(cluster_size)

df_scaled_with_clusters = pd.DataFrame(X_scaled, columns=X_features.columns)
df_scaled_with_clusters['Cluster'] = cluster_labels
df_scaled_with_clusters['Cluster'] = df_scaled_with_clusters['Cluster'].astype('category')
scaled_cluster_analysis = df_scaled_with_clusters.groupby('Cluster').mean(numeric_only=True)

plt.figure(figsize=(20, 8))
sns.heatmap(scaled_cluster_analysis, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title(f'Fingerprint dos {IDEAL_K} Regimes (Valores Padronizados)', fontsize=20)
plt.xlabel('Variáveis do Processo')
plt.ylabel('Cluster (Regime)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
save_current_figure('cluster_fingerprint_heatmap.png')
plt.show()

save_interim_data(cluster_analysis, 'cluster_analysis_results.csv')
save_interim_data(df_cluster_plot, 'cluster_labels_pca.csv')