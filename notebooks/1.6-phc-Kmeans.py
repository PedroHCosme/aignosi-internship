import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from aignosi_case.config import PLOT_FIGSIZE, SEABORN_STYLE
from aignosi_case.dataset import load_hourly_data, save_interim_data
from aignosi_case.plots import save_current_figure

sns.set_theme(style=SEABORN_STYLE, rc={'figure.figsize': PLOT_FIGSIZE})
df_hourly = load_hourly_data()

df_clean = df_hourly.dropna()
# Exclude target variables from clustering features
X_features = df_clean.drop(columns=['% Iron Concentrate', '% Silica Concentrate'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

# Elbow method: compute WCSS for k=1 to k=10
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
save_current_figure('kmeans_elbow_method.png')
plt.show()

IDEAL_K = 3

kmeans_final = KMeans(n_clusters=IDEAL_K, init='k-means++', n_init=10, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Project clusters onto 2D PCA space for visualization
pca_2 = PCA(n_components=2)
X_pca_2d = pca_2.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca_2d, columns=['PC1', 'PC2'], index=df_clean.index)
df_pca['Cluster'] = pd.Categorical(cluster_labels)

pc1_var = pca_2.explained_variance_ratio_[0] * 100
pc2_var = pca_2.explained_variance_ratio_[1] * 100

plt.figure(figsize=(14, 9))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='Set1', alpha=0.7, s=20)
plt.title(f'K-Means Clusters (K={IDEAL_K}) no Espaço PCA', fontsize=16)
plt.xlabel(f'PC1 ({pc1_var:.2f}%)')
plt.ylabel(f'PC2 ({pc2_var:.2f}%)')
plt.legend(title='Cluster')
plt.grid(True)
save_current_figure('kmeans_clusters_pca.png')
plt.show()

df_analysis = df_clean.copy()
df_analysis['Cluster'] = pd.Categorical(cluster_labels)

print("\n" + "="*50)
print(f"Análise dos {IDEAL_K} Regimes Operacionais")
print("="*50)

# Compute mean of all features grouped by cluster
cluster_analysis = df_analysis.groupby('Cluster').mean(numeric_only=True)

key_cols_analysis = ['% Silica Concentrate', '% Iron Concentrate', '% Silica Feed', 'Starch Flow', 'Amina Flow', 'Flotation Column 01 Air Flow']
print(cluster_analysis[key_cols_analysis])

# Compute normalized cluster size distribution
cluster_size = df_analysis['Cluster'].value_counts(normalize=True).sort_index()
print("\nDistribuição dos Clusters:")
print(cluster_size)

save_interim_data(df_analysis[['Cluster']], 'kmeans_cluster_labels.csv')
save_interim_data(cluster_analysis, 'kmeans_cluster_analysis.csv')