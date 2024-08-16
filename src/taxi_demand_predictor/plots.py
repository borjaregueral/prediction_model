import logging
import pandas as pd
from typing import List, Tuple, Optional
import plotly.express as px
import plotly.graph_objs as go
from datetime import timedelta


# Configure logging
logging.basicConfig(level=logging.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s' )

def plot_ts(df: pd.DataFrame, location: str = None, time_col: str = 'pickup_time', location_col: str = 'pickup_location_id', ride_count_col: str = 'ride_count') -> None:
    
    # Ensure the time column is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Filter the DataFrame for the specified location if provided
    if location is not None:
        df = df[df[location_col] == location]
    
    # Group by the time column and sum the ride counts
    df_grouped = df.groupby(time_col)[ride_count_col].sum().reset_index()
    
    # Plot the data using Plotly
    title = f'Rides by hour: Location {location}' if location is not None else 'Rides by hour: All Locations'
    fig = px.line(df_grouped, x=time_col, y=ride_count_col, template='plotly_dark', title=title)
    
    fig.show()



def plot_train_and_target(df: pd.DataFrame,
                          sample: int, 
                          target_column: str,
                          predictions: Optional[pd.DataFrame] = None,
                          location: Optional[int] = None) -> None:
    """
    Plots the train set and target data by column as a time series, with optional predictions.

    :param data_path: Path to the data parquet file.
    :param sample: Index of the row to be plotted.
    :param target_column: Name of the target column in the data.
    :param predictions: Optional DataFrame containing the predictions.
    :param location: Optional location ID to filter the data.
    """
    # Load the data
    #df = pd.read_parquet(data_path)
    # From df

    # Ensure the 'pickup_time' and target column exist
    if 'pickup_time' not in df.columns:
        raise KeyError("'pickup_time' column is missing from the data")
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' column is missing from the data")

    # Filter the DataFrame by location if provided
    if location is not None:
        df = df[df['pickup_location_id'] == location]
        if df.empty:
            raise ValueError(f"No data available for location {location}")

    # Extract time series columns in reverse order
    ts_columns = sorted([c for c in df.columns if c.startswith('rides_previous_')], reverse=True)

    # Select the specific row based on the sample index
    row = df.iloc[sample]
    ts_values = [row[c] for c in ts_columns]
    ts_dates = pd.date_range(
        end=row['pickup_time'] - timedelta(hours=1),
        periods=len(ts_columns),
        freq='h'
    )

    # Ensure the lengths of ts_dates and ts_values match
    if len(ts_dates) != len(ts_values):
        raise ValueError("The lengths of the time series dates and values do not match.")

    # Plot the train data
    title = f'Pickup date={row["pickup_time"].date()}, Pickup time={row["pickup_time"].time()}'
    if location is not None:
        title += f', Location={location}'
    else:
        title += f', Location={row["pickup_location_id"]}'
    
    fig = px.line(
        x=ts_dates, y=ts_values,
        template='plotly_dark',
        markers=True, title=title,
        labels={'x': 'Timestamp', 'y': 'Rides Count'}
    )

    # # Plot the target data
    target_value = row[target_column]
    target_date = row['pickup_time']
    fig.add_scatter(x=[target_date], y=[target_value],
                    line_color='red',
                    mode='markers+text', marker_symbol='x', marker_size=10,
                    name='Actual value')

    # Plot the predictions if provided
    if predictions is not None:
        prediction_value = predictions.iloc[sample] if isinstance(predictions, pd.Series) else predictions.iloc[sample, 0]
        fig.add_scatter(x=[target_date],
                        y=[prediction_value],
                        line_color='blue',
                        mode='markers', marker_symbol='circle', marker_size=10,
                        name='Prediction')

    #fig.show() # Commented for Streamlit although in the jupyter notebook should be included
    # Update the layout with a smaller font for the title
    fig.update_layout(
        title={
            'text': title,
            'font': {
                'size': 12  # Set the font size here
            }
        }
    )
    
    return fig