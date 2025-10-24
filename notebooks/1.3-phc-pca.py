from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# seaborn style settings
sns.set_theme(style="whitegrid", rc={'figure.figsize':(12,6)})

# load dataset and setting first column as datetime and index
csv_filename = 'MiningProcess_Flotation_Plant_Database.csv'
workspace_root = Path('/home/pedrocosme/aignosi/aignosi-case')  
data_path = workspace_root / 'data' / 'raw' / csv_filename

df = pd.read_csv(data_path, parse_dates=[0], index_col=0, decimal=',')


# downsampling to hourly frequency by aggregating 20-second data using mean over each hour
df_hourly = df.resample('h').mean()

# --- 1. Preparação dos Dados para PCA ---

# Criar um dataframe temporário para esta análise, removendo as 318 horas do shutdown
print(f"\nFormato original de df_hourly: {df_hourly.shape}")
df_pca_ready = df_hourly.dropna()
print(f"Formato para PCA (sem NaNs): {df_pca_ready.shape}")

# Separar nosso target (y) das variáveis independentes (X)
y_target_series = df_pca_ready['% Silica Concentrate']
X_features = df_pca_ready.drop(columns=['% Iron Concentrate', '% Silica Concentrate'])

# Scaling para o PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

print(f"\nRodando PCA em {X_scaled.shape[1]} features (variáveis)...")

# --- 2. Análise de Variância Explicada ---

# PCA com todos os componentes para ver a curva de cotovelo
pca_full = PCA()
pca_full.fit(X_scaled)

# Calcular a variância acumulada
explained_variance_cumulative = np.cumsum(pca_full.explained_variance_ratio_)

# Encontrar quantos componentes são necessários para 95% da variância
n_components_95 = np.where(explained_variance_cumulative >= 0.95)[0][0] + 1
print(f"\n--- Descoberta PCA (Redundância) ---")
print(f"São necessários {n_components_95} componentes para explicar 95% da variância.")
print(f"Conclusão: Conseguimos 'comprimir' {X_scaled.shape[1]} variáveis em {n_components_95}!")

# Plotar a Curva de Variância Explicada
plt.figure(figsize=(12, 7))
plt.plot(explained_variance_cumulative, marker='o', linestyle='--')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Explicada Cumulativa')
plt.title('Variância Explicada vs. Número de Componentes')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variância')
plt.axvline(x=n_components_95 - 1, color='g', linestyle='--', label=f'{n_components_95} Componentes')
plt.legend()
plt.show()

# --- 3. Visualização 2D (Análise de Regimes Operacionais) ---
print("\nGerando visualização 2D (PC1 vs PC2)...")

# Rodar o PCA novamente mas com apenas os 2 primeiros componentes
pca_2 = PCA(n_components=2)
X_pca_2d = pca_2.fit_transform(X_scaled)

df_pca_plot = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])
df_pca_plot['Silica (Alvo)'] = y_target_series.values 

# Explicar a variância dos 2 primeiros componentes
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
    palette='coolwarm',  # (Azul=baixa sílica, Vermelho=alta sílica)
    alpha=0.6,
    s=20 
)
plt.title(f'PCA (PC1 vs PC2) Colorido pela Concentração de Sílica (Alvo)\nPC1+PC2 explicam {pc1_var + pc2_var:.2f}% da variância', fontsize=16)
plt.xlabel(f'Componente Principal 1 ({pc1_var:.2f}%)')
plt.ylabel(f'Componente Principal 2 ({pc2_var:.2f}%)')
plt.legend(title='Sílica (%)')
plt.grid(True)
plt.show()