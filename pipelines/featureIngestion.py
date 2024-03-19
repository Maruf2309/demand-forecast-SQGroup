from zenml import pipeline
from zenml.steps.step_output import Output
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
from steps.load import load_data
import config
import pandas as pd


@pipeline(name='SQ Group Feature Ingestion Pipeline', enable_cache=False, enable_step_logs=True, enable_artifact_metadata=True)
def FeatureIngestionPipeline():
    """
    Pipeline that runs the data ingestion and data cleaning steps.
    """
    try:
        df = ingest_data(data_source=config.DATA_SOURCE)
        df = clean_data(data=df)
        targets, static_features, application_group, uses, mkt = encode_and_aggregate_data(data=df)
        features, target, imputer = merge_all_features(targets, static_features, application_group, uses, mkt)
        loaded = load_data(features=features, target=target)
    except Exception as e:
        logging.error(
            f"Error running data feature pipeline: {e}")

