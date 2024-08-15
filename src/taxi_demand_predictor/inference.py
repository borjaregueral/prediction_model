
from datetime import datetime, timedelta
import hopsworks
from hsfs.feature_store import FeatureStore
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import taxi_demand_predictor.config as cfg
import taxi_demand_predictor.data as data
import logging

logger = logging.getLogger(__name__)

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=cfg.HOPSWORKS_PROJECT_NAME,
        api_key_value=cfg.HOPSWORKS_API_KEY
    )

def get_feature_store():
    """Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """

    return get_hopsworks_project().get_feature_store()

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """"""
    # past_rides_columns = [c for c in features.columns if c.startswith('rides_')]
    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)
    # print(f'{results.shape = }')
    # print(f'{results.head()}')
    return results


def load_batch_of_features_from_store(current_date: pd.Timestamp ) -> pd.DataFrame:
    """Fetches the batch of features used by the ML system at `current_date`.

    Args:
        current_date (datetime): datetime of the prediction for which we want
        to get the batch of features

    Returns:
        pd.DataFrame: 4 columns:
            - `pickup_hour`
            - `rides`
            - `pickup_location_id`
            - `pickup_ts`
    """
    n_features = cfg.N_FEATURES

    feature_view = get_or_create_feature_view(cfg.FEATURE_VIEW_METADATA)

    # fetch data from the feature store
    fetch_data_to = cfg.END_DATE
    fetch_data_from = cfg.START_DATE

    # add plus minus margin to make sure we do not drop any observation
    ts_data = feature_view.get_batch_data(
        start_time=fetch_data_from - timedelta(days=1),
        end_time=fetch_data_to + timedelta(days=1)
    )

    # filter data to the time period we are interested in
    pickup_ts_from = int(fetch_data_from.timestamp() * 1000)
    pickup_ts_to = int(fetch_data_to.timestamp() * 1000)
    ts_data = ts_data[ts_data.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

    # sort data by location and time
    #ts_sorted = ts_data.sort_values(by=['pickup_location_id', 'pickup_time'], inplace=True)

    features = data.generate_training_set(ts_data, start_position = 0, 
                                          n_features=n_features, 
                                          step_size=cfg.STEP_SIZE, 
                                          pickup_location_id = None, 
                                          target_col = 'ride_count').drop(columns=['rides_next_hour'])
    
    return features
    

def load_model_from_registry():
    """
    Loads a machine learning model from the model registry.
    Returns:
        The loaded machine learning model.
    """
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name=cfg.MODEL_NAME,
        version=cfg.MODEL_VERSION,
    )  
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir)  / 'model.pkl')
       
    return model 

def load_predictions_from_store(from_pickup_hour: datetime,to_pickup_hour: datetime) -> pd.DataFrame:
    """
    Connects to the feature store and retrieves model predictions for all
    `pickup_location_id`s and for the time period from `from_pickup_hour`
    to `to_pickup_hour`

    Args:
        from_pickup_hour (datetime): min datetime (rounded hour) for which we want to get
        predictions

        to_pickup_hour (datetime): max datetime (rounded hour) for which we want to get
        predictions

    Returns:
        pd.DataFrame: 3 columns:
            - `pickup_location_id`
            - `predicted_demand`
            - `pickup_hour`
    """

    # get pointer to the feature view
    predictions_fv = get_or_create_feature_view(cfg.FEATURE_VIEW_PREDICTIONS_METADATA)

    # get data from the feature view
    print(f'Fetching predictions for `pickup_hours` between {from_pickup_hour}  and {to_pickup_hour}')
    predictions = predictions_fv.get_batch_data(
        start_time=from_pickup_hour - timedelta(days=1),
        end_time=to_pickup_hour + timedelta(days=1)
    )
    
    # make sure datetimes are UTC aware
    predictions['pickup_hour'] = pd.to_datetime(predictions['pickup_hour'], utc=True)
    from_pickup_hour = pd.to_datetime(from_pickup_hour, utc=True)
    to_pickup_hour = pd.to_datetime(to_pickup_hour, utc=True)

    # make sure we keep only the range we want
    predictions = predictions[predictions.pickup_hour.between(from_pickup_hour, to_pickup_hour)]

    # sort by `pick_up_hour` and `pickup_location_id`
    predictions.sort_values(by=['pickup_hour', 'pickup_location_id'], inplace=True)

    return predictions

def get_or_create_feature_view(feature_view_metadata):
    """"""

    # get pointer to the feature store
    feature_store = get_feature_store()

    # get pointer to the feature group
    # from src.config import FEATURE_GROUP_METADATA
    feature_group = feature_store.get_feature_group(
        name=feature_view_metadata['name'],
        version=feature_view_metadata['version']
    )

    # create feature view if it doesn't exist
    try:
        feature_store.create_feature_view(
            name=feature_view_metadata['name'],
            version=feature_view_metadata['version'],
            query=feature_group.select_all()
        )
    except:
        logger.info("Feature view already exists, skipping creation.")
    
    # get feature view
    feature_store = get_feature_store()
    feature_view = feature_store.get_feature_view(
        name=feature_view_metadata['name'],
        version=feature_view_metadata['version'],
    )

    return feature_view