from zenml import step
import pandas as pd
import logging, config
from typing import Union
from typing_extensions import Annotated


@step(name='Ingest Data', enable_cache=False, enable_step_logs=True)
def ingest_data(data_source: str) -> Annotated[pd.DataFrame, 'data']:
    """
    Ingests data from a given path.

    Args:
        data_source: The path to the data.

    Returns:
        The data as a string.
    """
    try:
        return pd.read_parquet(path=data_source)
    except Exception as e:
        logging.error(f"Error reading data from {config.DATA_SOURCE}: {e}")
        raise e
