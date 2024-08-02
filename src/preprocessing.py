import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def split_data(df: pd.DataFrame, datetime_column: str, cutoff_datetime: str, target_column: str) -> tuple:
    """
    Splits a DataFrame into training and testing sets based on a cutoff datetime.

    Parameters:
    df (pd.DataFrame: the input DataFrame file containing the data.
    datetime_column (str): The name of the column containing datetime values.
    cutoff_datetime (str or pd.Timestamp): The cutoff datetime for splitting the data.
    target_column (str): The name of the target column in the DataFrame.

    Returns:
    tuple: A tuple containing four elements:
        - X_train (pd.DataFrame): The training set features.
        - y_train (pd.Series): The training set target.
        - X_test (pd.DataFrame): The testing set features.
        - y_test (pd.Series): The testing set target.
    """
    # Ensure cutoff_datetime is a Timestamp
    cutoff_datetime = pd.Timestamp(cutoff_datetime)
    
    # Ensure the datetime and target columns exist
    if datetime_column not in df.columns:
        raise KeyError(f"'{datetime_column}' column is missing from the data")
    if target_column not in df.columns:
        raise KeyError(f"'{target_column}' column is missing from the data")
    
    # Split data based on cutoff datetime
    train_indices = df[df[datetime_column] < cutoff_datetime].index
    test_indices = df[df[datetime_column] >= cutoff_datetime].index
    
    X_train = df.loc[train_indices].drop(columns=[target_column])
    y_train = df.loc[train_indices, target_column]
    X_test = df.loc[test_indices].drop(columns=[target_column])
    y_test = df.loc[test_indices, target_column]
    
    return X_train, y_train, X_test, y_test

def period_avg(df: pd.DataFrame, period_length: int, time_unit: str ='hours', new_col_name: str ='period_avg') -> pd.DataFrame:
    """
    Adds a new column to the DataFrame that is the average of the specified period for the given time unit.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    period_length (int): The length of the period (e.g., 672 for 4 weeks in hours, 28 for 4 weeks in days).
    time_unit (str): The time unit for the period ('hours' or 'days'). Default is 'hours'.
    new_col_name (str): The name of the new column to be added. Default is 'period_avg'.

    Returns:
    pd.DataFrame: The DataFrame with the new column added.
    """
    # Generate the list of column names based on the time unit and period length
    if time_unit == 'hours':
        columns_to_average = [f'rides_previous_{i}_hour' for i in range(period_length, 0, -1)]
    elif time_unit == 'days':
        columns_to_average = [f'rides_previous_{i}_day' for i in range(period_length, 0, -1)]
    else:
        raise ValueError("Invalid time unit. Use 'hours' or 'days'.")
    
    # Calculate the row-wise average of these columns
    df[new_col_name] = df[columns_to_average].mean(axis=1)
    
    return df

class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        
        # Generate numeric columns from datetime
        X_["hour"] = X_['pickup_time'].dt.hour
        X_["day_of_week"] = X_['pickup_time'].dt.dayofweek
        
        return X_.drop(columns=['pickup_time'])
    
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop)