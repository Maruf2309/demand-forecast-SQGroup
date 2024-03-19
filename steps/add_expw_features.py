from zenml import step
import logging
import pandas as pd
from typing import Union, Annotated
from feature_engine.timeseries.forecasting import ExpandingWindowFeatures

logger = logging.getLogger(__name__)


@step(name='Generate Expanding Window Features', enable_step_logs=True, enable_artifact_metadata=True)
def AddExpWindowFeatures(targets: Annotated[pd.DataFrame, 'after added temporal features']) -> Annotated[pd.DataFrame, 'added Expanding Window features']:
    """Add Expanding Window Features to the data.
    Args:
        data (pd.DataFrame): The input data.
    Returns:
        pd.DataFrame: The data with added expanding window features.
    """
    try:

        expwindow = ExpandingWindowFeatures(
            variables=None, min_periods=7, functions='std', 
            periods=7, freq=None, sort_index=True, 
            missing_values='raise', drop_original=False
        )
        features = expwindow.fit_transform(targets[['timestamp', 'net_price', 'qtym']])
        
        # # 
        # for col in list(features.columns)[3:]:
        #     data[col] = features[col].values
        return features.drop(['timestamp', 'net_price', 'qtym'], axis=1)
    except Exception as e:
        logging.error(f'in The add_expw_features(): {e}')
        raise e
