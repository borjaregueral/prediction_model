import os
from dotenv import load_dotenv
from taxi_demand_predictor.paths import RAW_DATA_DIR, BRONZE_DATA_DIR, SILVER_DATA_DIR, GOLD_DATA_DIR, PARENT_DIR
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

FEATURE_GROUP_NAME = 'hourly_features_2023_2024'    
FEATURE_GROUP_VERSION = 1
