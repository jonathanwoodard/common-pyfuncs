import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pacmap
import plotly.express as px
import plotly.graph_objs as go
import hdbscan
import re

# 3D scatter plot function with labels
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
    fig.write_html('mushroom.html', auto_open=False)


# toy example using the kaggle mushrooms dataset
df = pd.read_csv('~/mushrooms.csv')
_df = pd.get_dummies(df[df.columns[1:]]).astype(int)
X = _df.to_numpy(copy=True) 
y = df['class'] 

# use PaCMAP to generate a 3D embedding
embedding = pacmap.PaCMAP(n_components=3, n_neighbors=10, MN_ratio=0.75, FP_ratio=5) 
X_transformed = embedding.fit_transform(X, init="random")
x = X_transformed[:, 0]  # values in the first dimension
y = X_transformed[:, 1]  # values in the second dimension
z = X_transformed[:, 2]  # values in the third dimension
# add xyz coordinates to df
df3d = df.copy() 
df3d['x'] = x
df3d['y'] = y
df3d['z'] = z
df3d = df3d.reset_index()

# assign cluster ids with HDBSCAN
_df = df3d[['x', 'y', 'z']].copy()
clusterer = hdbscan.HDBSCAN(min_cluster_size=120,min_samples=120,core_dist_n_jobs=-1).fit(_df)
df3d['cluster_id'] = clusterer.labels_
df3d['probability'] = clusterer.probabilities_
# adjust color saturation based on cluster probabilies
sat = np.where(clusterer.probabilities_>0.2,clusterer.probabilities_>0.2,0.2).reshape(-1,1)

# generate a custom colormap
colors = cm.viridis(np.linspace(0, 1, clusterer.cluster_persistence_.shape[0]))
cluster_colors = np.array([colors[col][:-1] if col >= 0 else (0.5, 0.5, 0.5) for col in clusterer.labels_])
# re-map cluster labels to sort clusters by size
counts = df3d[df3d.cluster_id>=0].cluster_id.value_counts(ascending=True).reset_index()
counts.columns = ['cluster_id','count']
map = counts.reset_index().set_index('cluster_id')['index'].to_dict()
id_values = df3d['cluster_id'].values
df3d['cluster_id'] = [map[v] if v in map.keys() else -1 for v in id_values]
_colors = np.array([colors[v][:-1] for v in df3d['cluster_id'].values])
color_sat = np.concatenate((_colors,sat),axis=1)
df3d['colors'] = list(color_sat)

# generate a label for each data point using features
df3d['index'] = df3d['index'].apply(lambda x: f'Data ID:  {x}')
df3d['class'] = df3d['class'].apply(lambda x: f'[e]dible/[p]oisonous:  {x}')
df3d['cap-shape'] = df3d['cap-shape'].apply(lambda x: f'cap-shape:  {x}')
df3d['cap-surface'] = df3d['cap-surface'].apply(lambda x: f'cap-surface:  {x}')
df3d['cap-color'] = df3d['cap-color'].apply(lambda x: f'cap-color:  {x}')
df3d['gill-attachment'] = df3d['gill-attachment'].apply(lambda x: f'gill-attachment:  {x}')
df3d['gill-spacing'] = df3d['gill-spacing'].apply(lambda x: f'gill-spacing:  {x}')
df3d['gill-size'] = df3d['gill-size'].apply(lambda x: f'gill-size:  {x}')
df3d['gill-color'] = df3d['gill-color'].apply(lambda x: f'gill-color:  {x}')
df3d['odor'] = df3d['odor'].apply(lambda x: f'odor:  {x}')
df3d['spore-print-color'] = df3d['spore-print-color'].apply(lambda x: f'spore-print-color:  {x}')
df3d['population'] = df3d['population'].apply(lambda x: f'population:  {x}')
df3d['habitat'] = df3d['habitat'].apply(lambda x: f'habitat:  {x}')
cols = ['index', 'class', 'cap-shape', 'cap-surface', 'cap-color', 
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'odor','spore-print-color','population','habitat','cluster_id']
df3d['text'] = df3d[cols].apply(lambda x: '<br>'.join([str(v) for v in x.values]),axis=1)

# generate interactive plot as html file
basic_3d(df3d)




