from zenml import step
from typing import Tuple, Annotated
import pandas as pd
import config
from sklearn.model_selection import train_test_split
import logging

@step(name="Split Data", enable_artifact_metadata=True, enable_artifact_visualization=True, enable_step_logs=True)
def split_data(
    data: Annotated[pd.DataFrame, 'features and Target'],
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[Annotated[pd.DataFrame, 'X_train'], Annotated[pd.DataFrame, 'X_test'], Annotated[pd.Series, 'y_train'], Annotated[pd.Series, 'y_test']]:
    """
    Split the data into train and test sets.

    Args:
        data (pd.DataFrame): The input data.
        test_size (float): The proportion of the data to include in the test set. Default is 0.2.
        random_state (int): The seed for the random number generator. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train and test sets.
    """
    logging.info("Splitting data...")
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state)
    logging.info("Data split successfully.")
    return train.drop(columns=['timestamp', 'net_price', 'qtym']), test.drop(columns=['timestamp', 'net_price', 'qtym']), train['qtym'], test['qtym']
