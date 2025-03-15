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
import xgboost as xgb
from xgboost import plot_importance, plot_tree


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


def modified_plot_importance(booster, figsize, **kwargs):
    """
    wrapper around xgboost plot_importance function to allow customization
    saves png to s3
    """
    title = kwargs['title']
    client = utils._s3_client('default')
    _pre = 's3_location'
    bucket = 's3_bucket'
    stamp = dt.strftime(dt.now(),'%Y%m%d%H%M%S')
    fig, ax = plt.subplots(1,1,figsize=figsize)
    img_name = '{}/importance_{}_{}.png'.format(_pre,title.split()[-1],stamp)
    xgb.plot_importance(booster=booster, ax=ax, **kwargs)
    img_data = BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight')
    r = utils._png_to_s3(img_data, img_name, client, bucket=bucket)

