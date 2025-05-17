
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

# Carrega os dados
df = pd.read_csv("ecommerce_estatistica.csv")

# Cria a aplicação
app = Dash(__name__)
app.title = "Dashboard E-commerce"

# Layout da aplicação
app.layout = html.Div(children=[
    html.H1("Dashboard de Estatísticas de E-commerce", style={'textAlign': 'center'}),

    dcc.Graph(
        id='grafico_dispersao',
        figure=px.scatter(df, x="Preço", y="Qtd_Vendidos_Cod",
                          title="Dispersão: Preço vs Quantidade Vendida",
                          trendline="ols")
    ),

    dcc.Graph(
        id='grafico_barras',
        figure=px.bar(df, x="Marca", y="Qtd_Vendidos_Cod",
                      title="Quantidade Vendida por Marca")
    ),

    dcc.Graph(
        id='grafico_pizza',
        figure=px.pie(df, names="Temporada", title="Distribuição por Temporada")
    )
])

# Executa o servidor na porta do Render
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
