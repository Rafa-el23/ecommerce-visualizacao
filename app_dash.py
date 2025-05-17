
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

# Carregar dados
df = pd.read_csv("ecommerce_estatistica.csv")
df['Preço'] = pd.to_numeric(df['Preço'], errors='coerce')
df = df.dropna()

# App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Gráfico de histograma
hist_fig = px.histogram(df, x="Preço", nbins=20, title="Distribuição de Preço dos Produtos")

# Gráfico de dispersão
scatter_fig = px.scatter(df, x="Preço", y="Qtd_Vendidos_Cod", title="Dispersão: Preço vs Quantidade Vendida")

# Mapa de calor
corr = df.corr(numeric_only=True)
heatmap_fig = px.imshow(corr, text_auto=True, title="Mapa de Calor das Correlações")

# Gráfico de barras (top 10 marcas)
top_marcas = df['Marca'].value_counts().head(10)
bar_fig = px.bar(x=top_marcas.values, y=top_marcas.index, orientation='h', title="Top 10 Marcas")

# Gráfico de pizza (temporada)
pizza_fig = px.pie(df, names='Temporada', title="Distribuição por Temporada")

# Gráfico de densidade (Preço)
precos = df['Preço'].dropna().values
kde = gaussian_kde(precos)
x_vals = np.linspace(precos.min(), precos.max(), 200)
y_vals = kde(x_vals)
dens_fig = go.Figure()
dens_fig.add_trace(go.Scatter(x=x_vals, y=y_vals, fill='tozeroy'))
dens_fig.update_layout(title="Densidade de Preço dos Produtos", xaxis_title="Preço", yaxis_title="Densidade")

# Gráfico de regressão
reg_fig = px.scatter(df, x="Preço", y="Qtd_Vendidos_Cod", trendline="ols", title="Regressão Linear: Preço vs Quantidade Vendida")

# Layout
app.layout = dbc.Container([
    html.H1("Dashboard - Visualização de Dados de E-commerce", className="my-4 text-center"),

    dcc.Graph(figure=hist_fig),
    dcc.Graph(figure=scatter_fig),
    dcc.Graph(figure=heatmap_fig),
    dcc.Graph(figure=bar_fig),
    dcc.Graph(figure=pizza_fig),
    dcc.Graph(figure=dens_fig),
    dcc.Graph(figure=reg_fig),

], fluid=True)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
