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


