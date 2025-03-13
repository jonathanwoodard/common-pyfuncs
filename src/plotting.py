import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D



def basic_3d(df):
    data = [go.Scatter3d(
        x=df['x'].values,
        y=df['y'].values,
        z=df['z'].values,
        hoverinfo='text',
        hoverlabel=dict(
            bgcolor='cornsilk'
            ),
        hovertext=df['text'],
        mode='markers',
        marker=dict(
            size=2,
            color=df['colors']
            )
        )]
    layout = go.Layout(
        hovermode= 'closest',
        showlegend= False
        )
    fig = go.Figure(data=data, layout=layout)
    fig.write_html('temp.html', auto_open=False)

