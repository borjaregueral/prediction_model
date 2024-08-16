import os
from datetime import datetime, timedelta
import logging
import pandas as pd
from typing import List
from taxi_demand_predictor.data import save_data, transform_data, add_missing_times
from taxi_demand_predictor.paths import RAW_DATA_DIR, BRONZE_DATA_DIR, SILVER_DATA_DIR, GOLD_DATA_DIR

def retrieve_data_in_range(start_timestamp: datetime, end_timestamp: datetime, base_path: str) -> pd.DataFrame:
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
    # Ensure start_date and end_date are datetime objects
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    #print(f"Base Path: {base_path}")


    file_paths = []
    
    for single_date in pd.date_range(start_date, end_date, freq='MS'):
        year = single_date.year
        month = single_date.month
        file_name = f"yellow_tripdata_{year}-{month:02d}.parquet"
        file_path = os.path.join(base_path, str(year), file_name)
        
        if os.path.exists(file_path):
            file_paths.append(file_path)
        else:
            print(f"File does not exist: {file_path}")
    
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

def validate_data(file_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Validates and saves the data for a specific year and month.

    :param file_path: The path of the file to be processed.
    :param start_date: The start date for filtering the data (inclusive).
    :param end_date: The end date for filtering the data (inclusive).
    :return: The validated DataFrame.
    """
    # Load the raw data
    rides = pd.read_parquet(file_path)
    
    # Convert the tpep_pickup_datetime column to datetime format
    rides['tpep_pickup_datetime'] = pd.to_datetime(rides['tpep_pickup_datetime'])
    
    # Define the start and end dates
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter the DataFrame to include only rows within the specified date range
    rides = rides[
        (rides['tpep_pickup_datetime'] >= start_date) &
        (rides['tpep_pickup_datetime'] <= end_date)
    ]
    
    # Select and rename columns, and optionally assign new columns
    rides = (
        rides[['tpep_pickup_datetime', 'PULocationID']]
        .rename(columns={'tpep_pickup_datetime': 'pickup_datetime', 'PULocationID': 'pickup_location_id'})
        .assign(pickup_time=lambda x: x['pickup_datetime'])  # Uncomment if needed
    )
    
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