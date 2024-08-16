import logging
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import List, Tuple
from taxi_demand_predictor.paths import RAW_DATA_DIR, BRONZE_DATA_DIR, SILVER_DATA_DIR, GOLD_DATA_DIR, create_directories

# Ensure directories are created
create_directories()

# Configure logging
logging.basicConfig(level=logging.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s' )

def download_file(url: str, download_path: Path):
    """
    Downloads a file from a URL to a specified path.

    :param url: The URL of the file to download.
    :param download_path: The path where the file should be saved.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an HTTPError for bad responses

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=download_path.name, colour='green')

    with open(download_path, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

    if total_size != 0 and t.n != total_size:
        logging.error(f"Error downloading {download_path.name}. Downloaded size does not match expected size.")

def download_data_files(base_url: str, year: int, months: list = None):
    """
    Downloads data files for a given year and optionally for specific months.

    :param base_url: The base URL where the data is hosted.
    :param year: The year of the data to fetch.
    :param months: The months of the data to fetch (optional, can be a list of integers).
    """
    months = list(range(1, 13)) if months is None else ([months] if isinstance(months, int) else months)
    
    # Create a folder within raw to store the data for each year
    year_folder = RAW_DATA_DIR / str(year)
    year_folder.mkdir(parents=True, exist_ok=True)
    
    file_names = [f"yellow_tripdata_{year}-{month:02d}.parquet" for month in months]
    urls = [f"{base_url}{file_name}" for file_name in file_names]
    download_paths = [year_folder / file_name for file_name in file_names]

    for url, download_path, file_name in zip(urls, download_paths, file_names):
        if download_path.exists():
            logging.info(f"File {file_name} already exists. Skipping download.")
            print(f"{file_name} skipped")
            continue

        logging.info(f"Downloading {file_name} from {url}")
        try:
            download_file(url, download_path)
            logging.info(f"Successfully downloaded {file_name}")
            print(f"{file_name} downloaded")
        except requests.HTTPError as e:
            if e.response.status_code == 403:
                logging.error(f"HTTP error occurred: {e}. Stopping downloads.")
                print(f"{file_name} doesn't exist")
                break
            else:
                logging.error(f"HTTP error occurred: {e}. Skipping this file.")
                continue
    
    print("All available files have been downloaded.")
    return [str(path.relative_to(RAW_DATA_DIR.parent)) for path in download_paths if path.exists()]

def join_parquet_files(input_folder: Path, output_folder: Path, output_file_name: str):
    """
    Joins all parquet files in the specified folder into a single parquet file and removes duplicates.
    The files are combined in order according to the month (01, 02, ..., 12).

    :param input_folder: The folder containing the parquet files to join.
    :param output_file_name: The name of the output parquet file.
    """
    parquet_files = sorted(input_folder.glob("*.parquet"), key=lambda x: x.stem)

    if not parquet_files:
        logging.info(f"No parquet files found in {input_folder}")
        return

    data_frames = []
    for file in tqdm(parquet_files, desc="Reading parquet files", colour='green'):
        df = pd.read_parquet(file)
        data_frames.append(df)

    combined_df = pd.concat(data_frames, ignore_index=True).drop_duplicates()

    output_file = output_folder / output_file_name
    combined_df.to_parquet(output_file)
    logging.info(f"Combined parquet file saved to {output_file}")

def save_data(rides: pd.DataFrame, folder: Path, file_name: str) -> None:
    """
    Saves the validated data to a parquet file.

    :param rides: The DataFrame containing the validated ride data.
    :param folder: The folder where the validated data will be saved.
    :param file_name: The name of the file to save the validated data.
    """
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)  # Create the year directory if it doesn't exist
        print(f'Folder "{folder}" created')
    save_path = folder / file_name
    
    # Save the DataFrame to a parquet file with a progress bar
    with tqdm(total=len(rides), desc="Saving data", unit="rows", colour='green') as pbar:
        rides.to_parquet(save_path)
        pbar.update(len(rides))
    
    print(f'Data saved to "{save_path}"')

def validate_and_save_data(file_path: str, year: int, month: int) -> pd.DataFrame:
    """
    Validates and saves the data for a specific year and month.

    :param file_path: The path of the file to be processed.
    :param year: The year of the data.
    :param month: The month of the data.
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
    
    # Filter data for the specific month
    start_time = pd.Timestamp(year, month, 1, 0, 0, 0)
    if month == 12:
        end_time = pd.Timestamp(year + 1, 1, 1, 0, 0, 0)
    else:
        end_time = pd.Timestamp(year, month + 1, 1, 0, 0, 0)
    rides = rides.loc[lambda x: (x['pickup_datetime'] >= start_time) & (x['pickup_datetime'] < end_time)]
    
    # Remove duplicates
    rides = rides.drop_duplicates()
    
    # Save the validated data
    validated_file_name = f"validated_yellow_tripdata_{year}-{month:02d}.parquet"
    save_data(rides, BRONZE_DATA_DIR / str(year), validated_file_name)
    
    return rides
    
def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    
        # Group by pickup_quarter_hour and pickup_location, and count the rides
    data_grouped = (
        data
        .assign(
            pickup_time=lambda df: df["pickup_datetime"].dt.floor('h').astype('datetime64[us]')
        )
    .groupby(['pickup_time', 'pickup_location_id'])
    .size()
    .reset_index(name='ride_count')
    )
    
    return data_grouped

# def add_missing_times(data: pd.DataFrame, freq: str) -> pd.DataFrame:
#     """
#     Adds missing times to the data based on the specified frequency.

#     :param data: The DataFrame containing the data.
#     :param freq: The frequency for adding missing times.
#     :return: The DataFrame with missing times added.
#     """
#     # Determine the start and end times
#     min_time = data['pickup_time'].min()
#     max_time = data['pickup_time'].max()
    
#     # Determine the start time as the first day of the month at 00:00:00
#     start_time = pd.Timestamp(min_time.year, min_time.month, 1).replace(hour=0, minute=0, second=0, microsecond=0)
    
#     # Determine the end time as the last data point in the original dataset
#     end_time = pd.Timestamp(max_time.year, max_time.month, 1).replace(hour=0, minute=0, second=0, microsecond=0)
#     # Determine the end time as the first day of the following month at 00:00:00
#     #end_time = (pd.Timestamp(max_time.year, max_time.month, 1) + pd.DateOffset(months=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
#     all_intervals = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive='left')
    
#     # Generate all possible locations
#     all_locations = data['pickup_location_id'].unique()
    
#     # Create a DataFrame with all combinations of intervals and locations
#     all_combinations = pd.MultiIndex.from_product([all_intervals, all_locations], names=['pickup_time', 'pickup_location_id'])
#     all_combinations_df = pd.DataFrame(index=all_combinations).reset_index()
    
#     # Merge with the original data to fill in missing times
#     merged_data = pd.merge(all_combinations_df, data, on=['pickup_time', 'pickup_location_id'], how='left')
    
#     # Fill NaN values with zero
#     merged_data = merged_data.fillna(0)
    
#     return merged_data

def add_missing_times(data: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Adds missing times to the data based on the specified frequency.

    :param data: The DataFrame containing the data.
    :param freq: The frequency for adding missing times.
    :return: The DataFrame with missing times added.
    """
    # Determine the start and end times
    min_time = data['pickup_time'].min()
    max_time = data['pickup_time'].max()

    # Determine the start time 
    start_time = min_time.floor(freq)

    # Determine the end time 
    end_time = max_time.ceil(freq)

    # Determine the end time as the first day of the following month at 00:00:00
    #end_time = (pd.Timestamp(max_time.year, max_time.month, 1) + pd.DateOffset(months=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    all_intervals = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive='left')
    
    # Generate all possible locations
    all_locations = data['pickup_location_id'].unique()
    
    # Create a DataFrame with all combinations of intervals and locations
    all_combinations = pd.MultiIndex.from_product([all_intervals, all_locations], names=['pickup_time', 'pickup_location_id'])
    all_combinations_df = pd.DataFrame(index=all_combinations).reset_index()
    
    # Merge with the original data to fill in missing times
    merged_data = pd.merge(all_combinations_df, data, on=['pickup_time', 'pickup_location_id'], how='left')
    
    # Fill NaN values with zero
    merged_data = merged_data.fillna(0)
    
    return merged_data

def transform_save_data_into_ts_data(data: pd.DataFrame, freq: str = 'h') -> pd.DataFrame:
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
    
    # # Extract year and month from the data
    # year = data['pickup_time'].dt.year.iloc[0]
    # unique_months = data['pickup_time'].dt.month.nunique()
    
    # # Define the transformed file name
    # if unique_months == 1:
    #     month = data['pickup_time'].dt.month.iloc[0]
    #     transformed_file_name = f"ts_data_{year}-{month:02d}.parquet"
    # else:
    #     transformed_file_name = f"ts_data_{year}.parquet"
    
    # Save the transformed data
    #save_data(transformed_data, Path(SILVER_DATA_DIR) / str(year), transformed_file_name)
    
    return transformed_data

def slice_and_slide(data, start_position, n_features, step_size, target_col):
    """
    Slices the data into features and targets using a sliding window approach.
    """
    indices_and_targets = []
    for start in range(start_position, len(data) - n_features - step_size + 1, step_size):
        end = start + n_features
        target = end + step_size
        indices_and_targets.append((start, end, target))
    return indices_and_targets

def process_location(data: pd.DataFrame, start_position: int, n_features: int, step_size: int, pickup_location_id: int, target_col: str = 'ride_count', message_printed: bool = False) -> pd.DataFrame:
    # Filter data for the specific pickup location
    data = data[data['pickup_location_id'] == pickup_location_id].copy()
    
    # Ensure 'pickup_time' is a datetime object
    data['pickup_time'] = pd.to_datetime(data['pickup_time'])
    
    # Ensure 'pickup_location_id' is an integer
    data['pickup_location_id'] = data['pickup_location_id'].astype('int32')

    # Generate indices and targets for slicing the data
    indices_and_targets = slice_and_slide(data, start_position, n_features, step_size, target_col=target_col)
    
    # Extract features (X) and targets (y)
    X = np.array([data.iloc[start:end][target_col].values for start, end, target in indices_and_targets])
    y = np.array([data.iloc[end:target][target_col].values[0] for start, end, target in indices_and_targets])

    # Extract year and month
    year = data['pickup_time'].dt.year.iloc[0]
    month = data['pickup_time'].dt.month.iloc[0]
    
    # Raise an error if no features are extracted
    if X.shape[0] == 0:
        if not message_printed:
            print(f'The number of features and step size are too big for {year}-{month}')
        return None  # Return None if no features are extracted

    # Extract additional columns for pickup time and location ID
    pickup_time_hours = [data.iloc[end:end+1]['pickup_time'].values[0] for start, end, target in indices_and_targets]
    pickup_location_ids = [data.iloc[end:end+1]['pickup_location_id'].values[0] for start, end, target in indices_and_targets]

    # Create a DataFrame with the extracted features and additional columns
    combined_df = pd.DataFrame(X, columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(n_features))])
    combined_df['pickup_time'] = pickup_time_hours
    combined_df['pickup_location_id'] = pickup_location_ids
    combined_df['rides_next_hour'] = y

    return combined_df

def generate_training_set(data: pd.DataFrame, start_position: int, n_features: int, step_size: int, pickup_location_id: int = None, target_col: str = 'ride_count') -> pd.DataFrame:
    
    
    data['pickup_time'] = pd.to_datetime(data['pickup_time'])
    year = data['pickup_time'].dt.year.iloc[0]
    month = data['pickup_time'].dt.month.iloc[0]

    message_printed = False

    if pickup_location_id is None:
        combined_dfs = []
        for loc_id in data['pickup_location_id'].unique():
            df = process_location(data, start_position, n_features, step_size, loc_id, target_col, message_printed)
            if df is not None:
                combined_dfs.append(df)
            else:
                message_printed = True
        if not combined_dfs:
            print(f"Skipping file for {year}-{month} due to insufficient data for all locations.")
            return None
        final_df = pd.concat(combined_dfs, ignore_index=True)
        filename = f"model_dataset_all_locations_{year}-{month}.parquet"
    else:
        final_df = process_location(data, start_position, n_features, step_size, pickup_location_id, target_col, message_printed)
        if final_df is None:
            print(f"Skipping file for location {pickup_location_id} in {year}-{month} due to insufficient data.")
            return None
        filename = f"model_dataset_location_{pickup_location_id}_{year}-{month}.parquet"

    final_df['pickup_time'] = final_df['pickup_time'].apply(lambda x: pd.Timestamp(x))
    final_df['pickup_location_id'] = final_df['pickup_location_id'].astype('int32')

    #save_data(final_df, Path(GOLD_DATA_DIR)/ str(year), filename)
    return final_df

def filter_by_location(df: pd.DataFrame, location: str, location_col: str = 'pickup_location_id') -> pd.DataFrame:
    # Filter the DataFrame for the specified location
    filtered_df = df[df[location_col] == location]
    return filtered_df

def slice_and_slide(data: pd.DataFrame, start_position: int, n_features: int, step_size: int, target_col: str = 'ride_count') -> List[Tuple[int, int, int]]:
    indices_and_targets = [
        (start, start + n_features, start + n_features + 1)
        for start in range(start_position, len(data) - n_features - 1, step_size)
        if (start + n_features + 1) < len(data)
    ]
    return indices_and_targets

