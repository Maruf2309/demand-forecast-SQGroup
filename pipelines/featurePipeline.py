from zenml import pipeline
from typing_extensions import Annotated, Tuple
import logging
from steps.ingest import ingest_data
from steps.clean import clean_data
from steps.encode import encode_and_aggregate_data
from steps.add_temporal_features import AddTemporalFeatures
from steps.add_lag_features import AddLagFeatures
from steps.add_window_features import AddWindowFeatures
from steps.add_expw_features import AddExpWindowFeatures
from steps.scale import scale_data
from steps.split import split_data
from steps.merge_all_feature import merge_all_features

import config
import pandas as pd


@pipeline(name='SQ Group ETL Pipeline')
def FeaturePipeline(data_source: Annotated[str, 'data_source']) -> Tuple[Annotated[pd.DataFrame, 'X_train'], Annotated[pd.DataFrame, 'X_test'], Annotated[pd.Series, 'y_train'], Annotated[pd.Series, 'y_test']]:
    """
    Pipeline that runs the data ingestion and data cleaning steps.
    """
    try:
        logging.info("Running Feature pipeline...")
        df = ingest_data(data_source=config.DATA_SOURCE)
        df = clean_data(data=df)
        targets, static_features, application_group, uses, mkt = encode_and_aggregate_data(data=df)
        features, target, imputer = merge_all_features(targets, static_features, application_group, uses, mkt)
        X_train, X_test, y_train, y_test = split_data(features=features, target=target)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(
            f"Error running data ingestion and cleaning pipeline: {e}")

