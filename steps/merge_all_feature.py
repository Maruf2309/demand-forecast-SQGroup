from zenml import step
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import BaseEstimator
from sklearn.impute import KNNImputer
from typing import Union
import logging
import numpy as np
from typing_extensions import Annotated, Tuple
from steps.add_temporal_features import AddTemporalFeatures
from steps.add_lag_features import AddLagFeatures
from steps.add_window_features import AddWindowFeatures
from steps.add_expw_features import AddExpWindowFeatures

@step(name='Merge All Features', enable_artifact_metadata=True, enable_step_logs=True)
def merge_all_features(
    targets: Annotated[pd.DataFrame, 'targets'], 
    static_features: Annotated[pd.DataFrame, 'static_features'],
    application_group: Annotated[pd.DataFrame, 'application_group'],
    uses: Annotated[pd.DataFrame, 'uses'],
    mkt: Annotated[pd.DataFrame, 'mkt'],
    
) -> Tuple[Annotated[pd.DataFrame, 'features'], Annotated[pd.Series, 'target'], Annotated[BaseEstimator, 'imputer']]:
    """Merges All Features into One.
    Args:
        data (pd.DataFrame): The input data.
    Returns:
        pd.DataFrame: The data with added expanding window features.
    """
    try:
        logging.info(f'==> merging features...')
        
        # Generate outlet wise timeseries_features
        timeseries_features_outlet_wise = pd.DataFrame()
        for outlet_id in targets.outlet_id.value_counts().index:
            outlet_wise = targets.loc[targets.outlet_id==outlet_id]
            temporal = AddTemporalFeatures(outlet_wise)
            lag_features = AddLagFeatures(outlet_wise)
            window_features = AddWindowFeatures(outlet_wise)
            exp_window_features = AddExpWindowFeatures(outlet_wise)
            outlet_wise_features = pd.concat([outlet_wise[['timestamp','outlet_id',]], temporal, lag_features, window_features, exp_window_features], axis=1)
            timeseries_features_outlet_wise = pd.concat([timeseries_features_outlet_wise, outlet_wise_features], ignore_index=True)
        
        # Merge outlet wise timeseries_features
        targets.merge(timeseries_features_outlet_wise, on=['timestamp','outlet_id',], how='inner')

        
        # Merge application group, uses, mkt
        data = targets.merge(
            application_group, on='outlet_id', how='inner'
        ).merge(
            uses, on='outlet_id', how='inner'
        ).merge(
            mkt, on='outlet_id', how='inner'
        ).merge(
            static_features, on=['timestamp', 'outlet_id'], how='inner'
        ).merge(
            timeseries_features_outlet_wise, on=['timestamp', 'outlet_id'], how='inner')
        
        # Impute Missing Values
        target = data['qtym']
        features = data.drop(columns=['timestamp', 'net_price', 'qtym', 'outlet_id'])
        imputer = KNNImputer(n_neighbors=5).fit(features)
        features = pd.DataFrame(imputer.transform(features), columns=features.columns)
        del data
        return features, target, imputer
    except Exception as e:
        logging.error(f'==> Error when merging features: {e}')
        raise e