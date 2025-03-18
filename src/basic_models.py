import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error as mse, r2_score as r2 
from sklearn import preprocessing as pre
import os
import re
import xgboost as xgb
from scipy.stats import skew
from datetime import datetime as dt, timedelta as td
import boto3
import logging, logging.config, logging.handlers
import hdbscan
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize, forest_minimize, gbrt_minimize, BayesSearchCV
from skopt.plots import plot_convergence,plot_evaluations,plot_objective
from skopt.callbacks import DeadlineStopper, VerboseCallback, DeltaXStopper
from sklearn.pipeline import Pipeline


"""
When dealing with complex data sets, visualizations can be very useful
for identifying relationships in the data. High dimensional data sets
are routinely reduced to 2 or 3 dimensions to make visualizations easier
to interpret. However, many commonly used algorithms (t-SNE, LargeVis, 
PacMAP, UMAP) are limited in the number of data points that can be 
reasonably processed.

In this application, a sample (50 - 100k records) from a large,  
high-dimensional data set is transformed to a 3D representation. HDBSCAN 
is used to cluster the data, and the 3D scatter plot can be used to assess 
the quality of the clustering. The assigned clusters can then be mapped 
back to the original data. This high-dimensional data with assigned and 
validated cluster ids can then be used as the training set for an XGBoost 
classification model to predict the cluster id for the original full data
set. 

As a validation step, a sample with predicted cluster ids can be run 
through the same process used to generate the training data. This allows 
the results of the two methods to be directly compared. 
"""


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


def calc_acceptance(met0, met1, met2):
    """
    The functions here are somewhat arbitrary. The idea is 
    to create curves shaped so that when the values for
    the three metrics approach the desired value, the sum
    of the values generated will approach some threshold
    """
    a = (np.sin(met0*np.pi)**1.7)*0.4
    a += (((np.cos(met1*np.pi)+1)/2)**3)*0.2
    a += (50*np.log((met2)/5)/(met2-1)**1.7)*0.5
    return np.max([a,0])


def clustering_iteration(df, mult, n=10):
    """
    This function uses the HDBSCAN algorithm to assign cluster ids
    to a low dimensional data set. The function uses a few simple
    metrics to determine if the clustering is high quality. These
    metrics are passed to a scoring function, and the clustering 
    can be repeated until an 'acceptable' score is reached.
    Metrics are:
        met0 - proportion of data points in the largest cluster
        met1 - proportion of data points not assigned to any cluster
        met2 - number of clusters identified
    The value of mult depends on the size of the dataframe
    """
    best_score = 0
    best_clust = None
    _iter = 0
    while _iter<n:
        clust_size = int((df.shape[0]*mult)+np.random.gamma(4.5,20)) # add randomness to iterations
        clusterer = hdbscan.HDBSCAN(min_cluster_size=clust_size,min_samples=clust_size,
                                    core_dist_n_jobs=-1).fit(df)
        df['cluster_id'] = clusterer.labels_
        met0 = df.cluster_id.value_counts().iloc[0]/df.shape[0]
        met1 = df[df.cluster_id==-1].shape[0]/df.shape[0]
        met2 = df['cluster_id'].max()
        score = calc_acceptance(met0, met1, met2)
        if (np.abs(0.5 - met0) < 0.15 and met1 < 0.1 and met2 > 7 and met2 <= 11) or score >= 1:
            best_score = score
            best_clust = clusterer
            _iter = n
        elif score > best_score:
            best_score = score
            best_clust = clusterer
            _iter += 1
        else:
            _iter += 1
    metrics = [met0, met1, met2, best_score]
    return metrics, best_clust, clusterer


def _train_classifer(df):
    """
    Create an XGBoost classifier and train on high dimensional data 
    with both categorical and continuous features
    """
    clf = xgb.XGBClassifier(base_score=0.2,colsample_bylevel=0.755,colsample_bytree=0.946,
                        gamma=0.0558,learning_rate=0.05,max_depth=5,n_estimators=200,n_jobs=-1,
                        objective='multi:softmax',num_class=num_class,random_state=42,reg_alpha=0.122,
                        reg_lambda=0.310,subsample=0.911,verbosity=1)
    clust_feat = ['cat_1', 'cat_2', 'cat_3', 'cat_4',
                  'cont_1', 'cont_2', 'cont_3']
    X_clust = pd.get_dummies(df[clust_feat],columns=clust_feat[:-3])
    # it's helpful to make sure features are in the same order and case
    X_clust = X_clust[X_clust.columns.sort_values()]
    lv_df,met0,met1,met2,clusterer = embed_and_cluster(X_clust)
    num_class = y.unique().shape[0]
    clf.set_params(num_class=num_class)
    clf.fit(X_train)
    return clf


def train_optimize_clf(X,y,bucket,prefix,key=None,n_calls=150):
    """
    Sample function for hyperparameter optimization with scikit-optimize
    https://scikit-optimize.github.io/stable/user_guide.html
    """
    num_class = y.unique().shape[0]
    today = dt.strftime(dt.today(),'%Y%m%d')
    # define model
    clf = xgb.XGBClassifier(base_score=0.2,colsample_bylevel=0.755,colsample_bytree=0.946,
                            gamma=0.0558,learning_rate=0.05,max_depth=5,n_estimators=200,n_jobs=-1,
                            objective='multi:softmax',num_class=num_class,random_state=42,reg_alpha=0.122,
                            reg_lambda=0.310,subsample=0.911,verbosity=1)
    # define hyperparameter space
    space = [Real(0.5,0.95,name='subsample'),
             Real(0.5,0.95,name='colsample_bytree'),
             Real(0.5,0.95,name='colsample_bylevel'),
             Real(0.01,5,'log-uniform',name='gamma'),
             Real(1e-2,10,name='reg_alpha'),
             Real(0.1,10,name='reg_lambda')]
    # define objective function for hyperparameter optimization
    @use_named_args(space)
    def objective(**params):
        mdl.set_params(**params)
        return 1-np.mean(cross_val_score(mdl, X, y, cv=5, n_jobs=-1, scoring='balanced_accuracy'))
    # optimize hyperparameters
    res_gb = gbrt_minimize(objective, space, n_calls=n_calls, random_state=42, n_jobs=-1, verbose=True,
                           callback=[DeadlineStopper(1800)])
    par = ['subsample','colsample_bytree','colsample_bylevel','gamma','reg_alpha','reg_lambda']
    _par = dict(zip(par,res_gb.x))
    mdl.set_params(**_par)
    mdl.set_params(n_estimators=500)
    mdl.fit(X,y)
    return mdl, res_gb
