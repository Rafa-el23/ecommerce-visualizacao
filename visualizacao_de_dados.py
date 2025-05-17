import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import numpy as np

# Configurações globais
plt.style.use('ggplot')
sns.set(font_scale=0.9)

# Leitura dos dados
df = pd.read_csv('ecommerce_estatistica.csv')
df['Preço'] = pd.to_numeric(df['Preço'], errors='coerce')
df = df.dropna()

# 1. Histograma
plt.figure(figsize=(8, 4))
sns.histplot(df['Preço'], bins=20)
plt.title('Distribuição de Preço dos Produtos')
plt.xlabel('Preço (R$)')
plt.ylabel('Quantidade de Produtos')
plt.tight_layout()
plt.show()

# 2. Gráfico de Dispersão
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Preço', y='Qtd_Vendidos_Cod')
plt.title('Dispersão entre Preço e Quantidade Vendida')
plt.xlabel('Preço (R$)')
plt.ylabel('Qtd. Vendida (Codificada)')
plt.tight_layout()
plt.show()

# 3. Mapa de Calor
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Mapa de Calor das Correlações Numéricas')
plt.tight_layout()
plt.show()

# 4. Gráfico de Barras (Top Marcas)
plt.figure(figsize=(10, 5))
top_marcas = df['Marca'].value_counts().head(10)
sns.barplot(x=top_marcas.values, y=top_marcas.index)
plt.title('Top 10 Marcas com Mais Produtos')
plt.xlabel('Número de Produtos')
plt.ylabel('Marca')
plt.tight_layout()
plt.show()

# 5. Gráfico de Pizza (Temporada)
plt.figure(figsize=(6, 6))
df['Temporada'].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
plt.title('Distribuição de Produtos por Temporada')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 6. Gráfico de Densidade com scipy
precos_array = df['Preço'].dropna().values
kde = gaussian_kde(precos_array)
x_vals = np.linspace(min(precos_array), max(precos_array), 200)
y_vals = kde(x_vals)

plt.figure(figsize=(8, 4))
plt.plot(x_vals, y_vals, color='blue')
plt.fill_between(x_vals, y_vals, alpha=0.4)
plt.title('Densidade de Preço dos Produtos')
plt.xlabel('Preço (R$)')
plt.ylabel('Densidade')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Gráfico de Regressão
plt.figure(figsize=(8, 5))
sns.regplot(x='Preço', y='Qtd_Vendidos_Cod', data=df, scatter_kws={'alpha': 0.5})
plt.title('Relação entre Preço e Quantidade Vendida (Regressão Linear)')
plt.xlabel('Preço (R$)')
plt.ylabel('Qtd. Vendida (Codificada)')
plt.tight_layout()
plt.show()
