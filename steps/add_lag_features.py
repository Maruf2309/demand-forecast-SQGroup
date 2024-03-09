from zenml import step
from dask import dataframe as dd
import logging
import pandas as pd
from typing import Union, Annotated
from feature_engine.timeseries.forecasting import LagFeatures

logger = logging.getLogger(__name__)


@step(name='Generate Lag Features', enable_step_logs=True, enable_artifact_metadata=True)
def AddLagFeatures(targets: Annotated[pd.DataFrame, 'after added temporal features']) -> Annotated[pd.DataFrame, 'Lag features']:
    """
    Add lag features to the data.
    """
    logging.info(f"Adding lag features to the data.")
    try:
        # Add Lag  Feature
        lagfeatures = LagFeatures(variables=None, periods=[3, 8, 16, 24], freq=None, sort_index=True,
                                  missing_values='raise', drop_original=False)
        lagfeatures.fit(targets[['timestamp', 'net_price', 'qtym']])
        features = lagfeatures.transform(
            targets[['timestamp', 'net_price', 'qtym']])
        # for col in list(features.columns)[3:]:
        #     data[col] = features[col].values
        logging.info(f'==> Successfully processed add_lag_features()')
        return features.drop(['timestamp', 'net_price', 'qtym'], axis=1)
    except Exception as e:
        logging.error(f'in The add_lag_features(): {e}')
        raise e
