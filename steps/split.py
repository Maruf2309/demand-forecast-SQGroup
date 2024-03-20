from zenml import step
from typing import Tuple, Annotated
import pandas as pd
import config
from sklearn.model_selection import train_test_split
import logging

@step(name="Split Data", enable_artifact_metadata=True, enable_artifact_visualization=True, enable_step_logs=True, enable_cache=False)
def split_data(
    data: Annotated[pd.DataFrame, 'features'],
    test_size: float = 0.25,
    random_state: int = 33
) -> Tuple[Annotated[pd.DataFrame, 'X_train'], Annotated[pd.DataFrame, 'X_test'], Annotated[pd.Series, 'y_train'], Annotated[pd.Series, 'y_test']]:
    """
    Split the data into train and test sets.

    Args:
        features (pd.DataFrame): The input data.
        target (pd.Series) : Target colun
        test_size (float): The proportion of the data to include in the test set. Default is 0.2.
        random_state (int): The seed for the random number generator. Default is 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train and test sets.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=['target']), data['target'], test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f'error in split_data : {e}')
        raise e
