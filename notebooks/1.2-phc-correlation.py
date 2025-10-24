from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# seaborn style settings
sns.set_theme(style="whitegrid", rc={'figure.figsize':(12,6)})

# load dataset and setting first column as datetime and index
csv_filename = 'MiningProcess_Flotation_Plant_Database.csv'
workspace_root = Path('/home/pedrocosme/aignosi/aignosi-case')  
data_path = workspace_root / 'data' / 'raw' / csv_filename

df = pd.read_csv(data_path, parse_dates=[0], index_col=0, decimal=',')


# downsampling to hourly frequency by aggregating 20-second data using mean over each hour
df_hourly = df.resample('h').mean()

# --- 1. Matriz de Correlação ---
plt.figure(figsize=(18, 9))
corr_matrix = df_hourly.corr()

# 1. Criar uma máscara para a diagonal superior
# np.triu -> "triangle upper"
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# 3. Desenhar o heatmap com a máscara e anotações menores
sns.heatmap(corr_matrix, 
            mask=mask,           # Aplicar a máscara
            annot=True,          # Manter anotações
            fmt='.2f',           # Formato de 2 casas decimais
            cmap='coolwarm',     # Mapa de cores
            vmin=-1, vmax=1,
            annot_kws={"size": 9}  # Diminui o tamanho da fonte da anotação
           ) 
plt.title('Matriz de Correlação', fontsize=12)  # Reduzir tamanho do título
plt.xticks(rotation=45, ha='right', fontsize=12)  # Reduzir tamanho dos labels do eixo X
plt.yticks(rotation=0, fontsize=12)  # Reduzir tamanho dos labels do eixo Y
plt.tight_layout() # Ajusta o layout para não cortar os labels
plt.show()

# --- Análise de Correlação no Terminal ---

# Pergunta 1: Quais variáveis mais afetam nosso alvo?
# Vamos focar no '% Silica Concentrate'
print("\n" + "="*50)
print("Correlações com o Alvo: '% Silica Concentrate'")
print("="*50)

# Pega a coluna do alvo, remove a correlação dela consigo mesma (1.00)
# e ordena em ordem decrescente
target_corr = corr_matrix['% Silica Concentrate'].drop('% Silica Concentrate')
print(target_corr.sort_values(ascending=False))

# Pergunta 2: Quais variáveis de entrada são redundantes?
# (Multicolinearidade: correlação > 0.8 ou < -0.8)
print("\n" + "="*50)
print("Pares de Alta Correlação (Multicolinearidade)")
print("="*50)

# Usamos a máscara que criamos para o plot
# .where() mantém os valores da diagonal inferior, o resto vira NaN
masked_corr = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))

# .stack() "empilha" a matriz removendo os NaNs
# Isso nos dá uma lista de pares únicos (A, B)
stacked_corr = masked_corr.stack()

# Filtra apenas pelas correlações absolutas muito altas
high_corr_pairs = stacked_corr[stacked_corr.abs() > 0.8]

if high_corr_pairs.empty:
    print("Nenhum par de variável com correlação > 0.8 encontrado.")
else:
    print(high_corr_pairs.sort_values(ascending=False))