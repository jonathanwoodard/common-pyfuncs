import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import pacmap
import boto3
import hdbscan
import tempfile


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


def plotly_scatter_3d(df,bucket,client,key,prefix):
    """
    This function takes data with multiple explanatory features
    that can be added to a label for each point in the plot.
    The plot is generated using the `basic_3d` function as an 
    interactive html file, which can be saved to s3
    """
    acl = {'ACL':"bucket-owner-full-control"}
    df['data_id'] = df.data_id.apply(lambda x: f'Data ID:  {x}')
    df['numerical_feature_1'] = df.numerical_feature_1.round(2).apply(lambda x: f'Numerical Feature 1:  {x}')
    df['string_feature_1'] = df.string_feature_1.apply(lambda x: f'String Feature 1:  {x}')
    df['list_feature'] = [f'Value:  {v}' for v in df['list_feature'].values]
    df['string_feature_2'] = df.string_feature_2.apply(lambda x: f'String Feature 2:  {x}')
    df['cluster_id'] = df.cluster_id.apply(lambda x: f'Cluster ID:  {x}')
    # concatenated feature values show up on hover
    cols = ['data_id','numerical_feature_1','string_feature_1','list_feature',
            'string_feature_2','cluster_id']
    df['text'] = df[cols].apply(lambda x: '<br>'.join([str(v) for v in x.values]),axis=1)
    basic_3d(df)
    _key = '/'.join([prefix,key])
    response = client.upload_file(Filename='temp.html',Bucket=bucket,Key=_key,ACL=acl)
    return response


def pacmap_embed(df,nc=3,nn=70,mn=0.5,fp=2):
    """
    This function uses the PacMAP package to embed a high-dimensional
    data set in an (x, y, z) coordinate system for plotting using the
    functions above. The 3D dataframe can be merged with the original
    dataframe to provide values for the plot's `text` field.
    PacMAP input and output data is numpy array format
    """
    # can use a subset of df columns
    X = df[df.columns[:-1]].to_numpy(copy=True) 
    embedding = pacmap.PaCMAP(n_components=nc, n_neighbors=nn, MN_ratio=mn, FP_ratio=fp)  
    t0 = dt.now()
    X_transformed = embedding.fit_transform(X, init="random") 
    print(f'Elapsed: {dt.now() - t0}')
    # The transformed data X_transformed is split into its components: x, y, and z
    x = X_transformed[:, 0]  # values in the first dimension
    y = X_transformed[:, 1]  # values in the second dimension
    z = X_transformed[:, 2]  # values in the third dimension
    # add xyz coordinates to df
    df3d = df.copy() 
    df3d['x'] = x
    df3d['y'] = y
    df3d['z'] = z
    return embedding, df3d
