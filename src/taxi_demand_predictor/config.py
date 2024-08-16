import os
from dotenv import load_dotenv
from taxi_demand_predictor.paths import RAW_DATA_DIR, BRONZE_DATA_DIR, SILVER_DATA_DIR, GOLD_DATA_DIR, PARENT_DIR
from datetime import datetime, timedelta
import pandas as pd

STREAM_DATA_DIR_RAW = RAW_DATA_DIR / "2023-2024"
STREAM_DATA_DIR_BRONZE = BRONZE_DATA_DIR / "2023-2024"
STREAM_DATA_DIR_SILVER = SILVER_DATA_DIR / "2023-2024"
STREAM_DATA_DIR_GOLD = GOLD_DATA_DIR / "2023-2024"
# INPUT_RAW_DIR = RAW_DATA_DIR / "2023-2024"/f'yellow_tripdata_{start.year}-{start.month:02d}_to_{end.year}-{end.month:02d}.parquet'
# INPUT_SILVER_DIR = STREAM_DATA_DIR_SILVER /f'ts_data_{start.year}-{start.month:02d}_to_{end.year}-{end.month:02d}.parquet'


# Path to the root directory of the project in hopsworks
HOPSWORKS_PROJECT_NAME = 'demand_predictor_borja'

# Load the environment variables
load_dotenv(PARENT_DIR / '.env')    

HOPSWORKS_API_KEY = os.environ.get('HOPSWORKS_API_KEY')

# number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28

# FEATURE_GROUP_NAME = 'hourly_features_2023_2024'    
# FEATURE_GROUP_VERSION = 1
# FEATURE_VIEW_NAME = 'hourly_features_view_2023_2024' 
# FEATURE_VIEW_VERSION = 1

FEATURE_VIEW_METADATA = {
    'name': 'view_group_features_hourly_2023_03_29_to_2024_05_24',
    'version': 1,
    'description': 'Feature view of historical taxi rides',
    'primary_key': ['pickup_location_id', 'pickup_ts'],
    'event_time': 'pickup_ts',
    'online_enabled': True,
}

FEATURE_GROUP_METADATA = {
    'name': 'group_features_hourly_2023_03_29_to_2024_05_24',
    'version': 1,
    'description': 'Feature group with hourly time-series data of historical taxi rides',
    'primary_key': ['pickup_location_id','pickup_ts'],
    'event_time': 'pickup_ts',
    'online_enabled': True,
}



# Model name and version in Hopsworks
MODEL_NAME = 'taxi_demand_predictor'
MODEL_VERSION = 4

# Number of historical values our model needs to generate predictions
N_FEATURES = 24 * 28
STEP_SIZE = 23

CURRENT_DATE = pd.to_datetime(datetime.utcnow()).floor('h')

END_DATE = CURRENT_DATE - timedelta(days=7*12)
START_DATE = END_DATE - timedelta(days=7*8)

# Number of steps in the frontend
N_STEPS = 7
# Number of tieseries destinations to show in the frontend
DESTINATIONS_TIMES = 3