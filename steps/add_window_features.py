from zenml import step
import logging
import pandas as pd
from typing import Union, Annotated
from feature_engine.timeseries.forecasting import WindowFeatures

logger = logging.getLogger(__name__)


@step(name='Generate Window Features', enable_step_logs=True, enable_artifact_metadata=True)
def AddWindowFeatures(targets: Annotated[pd.DataFrame, 'After lag features added']) -> Annotated[pd.DataFrame, 'window features']:
    """Add window features to the dataframe

    Args:
        data (Union[dd.DataFrame, pd.DataFrame]): The dataframe to add window features to.

    Returns:
        Union[dd.DataFrame, pd.DataFrame]: The dataframe with window features added.
    """

    try:
        windowfeatures = WindowFeatures(variables=None, window=24, freq=None, sort_index=True,
                                        missing_values='raise', drop_original=False)
        windowfeatures.fit(
            targets[['timestamp', 'net_price', 'qtym']])
        features = windowfeatures.transform(
            targets[['timestamp', 'net_price', 'qtym']])
        # for col in list(features.columns)[3:]:
        #     data[col] = features[col].values
        return features.drop(['timestamp', 'net_price', 'qtym'], axis=1)
    except Exception as e:
        logging.error(f'in add_window_features(): {e}')
        raise e
