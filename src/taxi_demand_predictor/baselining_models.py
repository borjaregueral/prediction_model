import numpy as np
import pandas as pd

class BaselineModelPreviousHour:
    """
    predict: Actual demand observed at last day and same hour
    """
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    def predict(self, X_test: pd.DataFrame)-> np.ndarray:
        return X_test['rides_previous_1_hour']
    
class BaselineModelPreviousWeek:
    """
    predict: Actual demand observed at t = 7 days
    """
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass
    def predict(self, X_test: pd.DataFrame)-> np.ndarray:
        return X_test[f'rides_previous_{24*7}_hour']
    
class BaselineModelLast4Weeks:
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass
    def predict(self, X_test: pd.DataFrame)-> np.ndarray:
        return 0.25*(X_test[f'rides_previous_{24*7}_hour'] + 
                    X_test[f'rides_previous_{24*7*2}_hour'] + 
                    X_test[f'rides_previous_{24*7*3}_hour'] + 
                    X_test[f'rides_previous_{24*7*4}_hour'])