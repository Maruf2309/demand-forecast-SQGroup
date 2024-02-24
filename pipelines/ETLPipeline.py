from zenml import pipeline
from typing_extensions import Annotated
import logging
from steps.ingest import ingest_data
from steps.clean import clean_data
from steps.encode import encode_data
from steps.add_temporal_features import AddTemporalFeatures
from steps.add_lag_features import AddLagFeatures
from steps.add_window_features import AddWindowFeatures
from steps.add_expw_features import AddExpWindowFeatures
from steps.scale import scale_data

import config
import pandas as pd


@pipeline(name='SQ Group ETL Pipeline')
def ETLFeaturePipeline(data_source: Annotated[str, 'data_source']) -> Annotated[pd.DataFrame, 'features']:
    """
    Pipeline that runs the data ingestion and data cleaning steps.
    """
    global ingest_data, clean_data
    try:
        logging.info("Running Feature pipeline...")
        data = ingest_data(config.DATA_SOURCE)
        data = clean_data(data)
        data = encode_data(data)
        temporalFeatures = AddTemporalFeatures(data)
        lagFeatures = AddLagFeatures(data)
        windowFeatures = AddWindowFeatures(data)
        expWindowFeatures = AddExpWindowFeatures(data)
        data = scale_data(data, temporalFeatures, lagFeatures, windowFeatures, expWindowFeatures)
        del expWindowFeatures, windowFeatures, lagFeatures, temporalFeatures
        return data
    except Exception as e:
        logging.error(
            f"Error running data ingestion and cleaning pipeline: {e}")

