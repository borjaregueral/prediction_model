import pandas as pd
from taxi_demand_predictor.paths import *
from typing import Callable
from taxi_demand_predictor.preprocessing import split_data, period_avg, TemporalFeaturesEngineer, ColumnDropper
import taxi_demand_predictor.config as cfg
from datetime import datetime
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit
import lightgbm as lgb
import optuna
import numpy as np

# Define a wrapper function for period_avg to use with FunctionTransformer
def period_avg_4_weeks(df):
    return period_avg(df, cfg.N_FEATURES, 'hours', '4_weeks_avg')

def get_pipeline(**hyperparameters):
    """
    Returns a pipeline object that preprocesses the data and fits a model.
    """

    # Create a FunctionTransformer
    add_4_weeks_avg = FunctionTransformer(period_avg_4_weeks, validate=False)
    
    # Create a TemporalFeaturesEngineer object
    tfe = TemporalFeaturesEngineer()
    
    # Create a ColumnDropper object
    cd = ColumnDropper(columns_to_drop=['pickup_location_id'])  
    
    # Create pipeline with the different steps involved in the preprocessing and model
    pipeline = make_pipeline(
        add_4_weeks_avg,
        tfe,
        cd,
        lgb.LGBMRegressor(**hyperparameters)
    )
    
    return pipeline

def objective(trial: optuna.trial.Trial, X_train, y_train, pipeline: Callable, metric: Callable = mean_absolute_error, n_splits: int = 3) -> float:
    """
    Given a set of hyper-parameters, it trains a model and computes an average
    validation error based on a KFold split.
    
    Parameters:
    - trial: optuna.trial.Trial
    - X_train: Training features
    - y_train: Training labels
    - get_pipeline: Function to get the model pipeline
    - metric: Evaluation metric function (default is mean_absolute_error)
    - n_splits: Number of splits for cross-validation (default is 3)
    
    Returns:
    - Mean validation score
    """
    # pick hyper-parameters
    hyperparams = {
        "metric": 'mae',
        "verbose": -1,
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 100),   
    }
       
    tss = KFold(n_splits=n_splits)
    scores = []
        
    for train_index, val_index in tss.split(X_train):

        # split data for training and validation
        X_train_, X_val_ = X_train.iloc[train_index, :], X_train.iloc[val_index,:]
        y_train_, y_val_ = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # train the model
        pipeline = get_pipeline(**hyperparams)
        pipeline.fit(X_train_, y_train_)
        
        # evaluate the model
        y_pred = pipeline.predict(X_val_)
        score = metric(y_val_, y_pred)

        scores.append(score)
   
    # Return the mean score
    return np.array(scores).mean()
