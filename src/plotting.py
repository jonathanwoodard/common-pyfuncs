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
import general_s3 as utils
import slack_alert as slert


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


def modified_tree_plot(df,tree,ax,logger,cmap='viridis'):
    """
    Modified version of condensed tree plot from HDBSCAN
    Clusters are sorted by size, which allows the numbering
    and colors to align with the 3D scatter plot
    Labelling of clusters modified for clarity
    """
    plot_data = tree.get_plot_data(leaf_separation=1.5)
    # need to increase bar_widths so that all the bars show up in the plot
    plot_data['_bar_widths'] = [w+5000 for w in plot_data['bar_widths']]
    if cmap != 'none':
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(0, max(plot_data['bar_widths'])))
        sm.set_array(plot_data['_bar_widths'])
        bar_colors = [sm.to_rgba(x) for x in plot_data['bar_widths']]
    else:
        bar_colors = 'gray'
    ax.bar(
        plot_data['bar_centers'],
        plot_data['bar_tops'],
        bottom=plot_data['bar_bottoms'],
        width=plot_data['_bar_widths'],
        color=bar_colors,
        align='center',
        linewidth=0.8
    )
    drawlines = []
    for xs, ys in zip(plot_data['line_xs'], plot_data['line_ys']):
        drawlines.append(xs)
        drawlines.append(ys)
    ax.plot(*drawlines, color='k', linewidth=1, alpha=1) # make lines bolder so they show up
    cluster_bounds = np.array([plot_data['cluster_bounds'][c] for c in df['child']])
    if not np.isfinite(cluster_bounds).all():
        msg = """
              Infinite lambda values encountered in chosen clusters.\n
              This might be due to duplicates in the data.
              """
        logger.warning(msg)
        r = slert.slack_alert(msg,'high')
    plot_range = np.hstack([plot_data['bar_tops'], plot_data['bar_bottoms']])
    plot_range = plot_range[np.isfinite(plot_range)]
    mean_y_center = np.mean([np.max(plot_range)*1.1, np.min(plot_range)])
    max_height = np.diff(np.percentile(plot_range, q=[10,90]))
    for i, c in enumerate(df['child']):
        c_bounds = plot_data['cluster_bounds'][c]
        # increase the width of the elipses so the bars aren't obscured
        width = ((c_bounds[1] - c_bounds[0]) + 4000)*5
        height = (c_bounds[3] - c_bounds[2]) * 1.2
        center = (
            np.mean([c_bounds[0], c_bounds[1]]),
            np.mean([c_bounds[3], c_bounds[2]]),
        )
        if not np.isfinite(center[1]):
            center = (center[0], mean_y_center)
        if not np.isfinite(height):
            height = max_height
        min_height = 0.1*max_height
        if height < min_height:
            height = min_height
        if cmap != 'none':
            selection_palette = hexcolormap(cmap,df.shape[0])
            oval_color = selection_palette[i]
        else:
            oval_color = '#77bd22'
        box = matplotlib.patches.Ellipse(
            center,
            width,
            height,
            facecolor='none',
            edgecolor=oval_color,
            linewidth=(1.2+i/4)
        )
        ax.annotate(str(i), xy=center,
                    xytext=(center[0]+9000, center[1]+0.03),
                    horizontalalignment='left',
                    verticalalignment='bottom',fontsize=16)
        ax.add_artist(box)
    cb = plt.colorbar(sm, ax=ax)
    cb.ax.set_ylabel('Number of points')
    ax.set_xticks([])
    for side in ('right', 'top', 'bottom'):
        ax.spines[side].set_visible(False)
    ax.invert_yaxis()
    ax.set_ylabel('$\lambda$ value')
    return ax


def hexcolormap(map,n):
    """
    Helper function to generate a list of hex color codes 
    from an mpl colormap. Used in the modified tree plot above.
    """
    _cm = cm.get_cmap(map)
    colors = _cm(np.linspace(0, 1, n))
    rgbmap = np.array([colors[col][:-1] for col in range(n)])
    hexmap = [('#'+''.join([f'{int(i):02x}' for i in 255*c])) for c in rgbmap]
    return hexmap
