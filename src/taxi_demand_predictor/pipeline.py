import os
from datetime import datetime, timedelta
import logging
import pandas as pd
from typing import List
from taxi_demand_predictor.data import save_data, transform_data, add_missing_times
from taxi_demand_predictor.paths import RAW_DATA_DIR, BRONZE_DATA_DIR, SILVER_DATA_DIR, GOLD_DATA_DIR

def retrieve_data_in_range(start_timestamp: str, end_timestamp: str, base_path: str) -> pd.DataFrame:
    """
    Retrieve and concatenate parquet files within a specified date range.

    Parameters:
    start_timestamp (str): The start timestamp in 'YYYY-MM-DD' format.
    end_timestamp (str): The end timestamp in 'YYYY-MM-DD' format.
    base_path (str): The base directory path containing year subfolders with parquet files.

    Returns:
    pd.DataFrame: A concatenated DataFrame containing data from all parquet files in the specified range.
    """
    start_date = pd.to_datetime(start_timestamp)
    end_date = pd.to_datetime(end_timestamp)
    
    file_paths = _get_file_paths_in_range(start_date, end_date, base_path)
    concatenated_df = _read_and_concat_parquet_files(file_paths)
    
    return concatenated_df

def _get_file_paths_in_range(start_date: datetime, end_date: datetime, base_path: str) -> List[str]:
    """
    Get the file paths for parquet files within the specified date range.

    Parameters:
    start_date (datetime): The start date.
    end_date (datetime): The end date.
    base_path (str): The base directory path containing year subfolders with parquet files.

    Returns:
    list: A list of file paths for the parquet files in the specified range.
    """
    file_paths = []
    
    for single_date in pd.date_range(start_date, end_date, freq='MS'):
        year = single_date.year
        month = single_date.month
        file_name = f"yellow_tripdata_{year}-{month:02d}.parquet"
        file_path = os.path.join(base_path, str(year), file_name)
        
        if os.path.exists(file_path):
            file_paths.append(file_path)
    
    return file_paths

def _read_and_concat_parquet_files(file_paths: List[str]) -> pd.DataFrame:
    """
    Read and concatenate parquet files from the given file paths.

    Parameters:
    file_paths (list): A list of file paths for the parquet files.

    Returns:
    pd.DataFrame: A concatenated DataFrame containing data from all parquet files.
    """
    dataframes = [pd.read_parquet(file_path) for file_path in file_paths]
    concatenated_df = pd.concat(dataframes, ignore_index=True)
    
    return concatenated_df

def validate_and_save_data(file_path: str, start_timestamp: str, end_timestamp: str) -> pd.DataFrame:
    """
    Validates and saves the data within a specified date range.

    :param file_path: The path of the file to be processed.
    :param start_timestamp: The start timestamp in 'YYYY-MM-DD' format.
    :param end_timestamp: The end timestamp in 'YYYY-MM-DD' format.
    :param save_dir: The directory where the validated data will be saved.
    :return: The validated DataFrame.
    """
    # Load the raw data
    rides = pd.read_parquet(file_path)
    
    # Validate the data
    rides = (
        rides[['tpep_pickup_datetime', 'PULocationID']]
        .rename(columns={'tpep_pickup_datetime': 'pickup_datetime', 'PULocationID': 'pickup_location_id'})
        .assign(pickup_time=lambda x: x['pickup_datetime'])
    )
    
    # Convert timestamps to pd.Timestamp
    start_time = pd.to_datetime(start_timestamp)
    end_time = pd.to_datetime(end_timestamp)
    
    # Filter data for the specific date range
    rides = rides.loc[lambda x: (x['pickup_datetime'] >= start_time) & (x['pickup_datetime'] < end_time)]
    
    # Remove duplicates
    rides = rides.drop_duplicates()
    
    return rides

def transform_to_ts_data(data: pd.DataFrame, freq: str = 'h') -> pd.DataFrame:
    """
    Transforms and saves the data into time series data.

    :param data: The DataFrame containing the data to be transformed.
    :param freq: The frequency for adding missing times (default is 'h' for hourly).
    :return: The transformed DataFrame.
    """
    # Transform the data
    data_grouped = transform_data(data)
    
    # Add missing times
    transformed_data = add_missing_times(data_grouped, freq)

    return transformed_data