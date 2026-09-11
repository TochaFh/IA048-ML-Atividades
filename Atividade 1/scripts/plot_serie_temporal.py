from pathlib import Path
import pandas as pd
import plotly.express as px


DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "air traffic.csv"


df = pd.read_csv(DATASET_PATH, thousands=',')

df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))

fig = px.line(
    df,
    x='Date',
    y='Flt',
    title='Série Temporal Completa - 2003 a 2023',
    labels={'Date': 'Ano', 'Flt': 'Número de Voos (Flt)'}
)

fig.update_traces(line=dict(width=5))

fig.update_xaxes(
    dtick="M12",            # Define o intervalo em meses
    tickformat="%Y",        # Garante que o rótulo mostre apenas o Ano
    tickangle=-45,          # Rotaciona os anos em -45° para que não fiquem sobrepostos
    hoverformat="%B %Y",    # Mostra o Mês inteiro e o Ano ao passar o mouse
    tick0="2003-01-01"      # Define o ano inicial exato para alinhar as marcações
)

fig.update_layout(font=dict(size=24))

fig.show()
