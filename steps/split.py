from zenml import step
from sklearn.model_selection import TimeSeriesSplit
import logging


# Define the split_data step
@step(enable_cache=True)
def split_data(data):
    """
    Split the time series into train, validation, and test sets.
    """
    try:
        logging.info("Splitting time series into train, validation, and test sets.")
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, val_idx, test_idx = next(tscv.split(data))
        logging.info("Splitting complete.")
        return data.iloc[train_idx], data.iloc[val_idx], data.iloc[test_idx]
    except Exception as e:
        logging.error(f"Error splitting time series: {e}")
        return None, None, None

    
