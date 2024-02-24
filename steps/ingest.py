from zenml import step
import pandas as pd
import logging
from typing import Union
from typing_extensions import Annotated


@step(name='Ingest Data')
def ingest_data(data_source: Annotated[str, 'data_source']) -> Annotated[pd.DataFrame, 'data']:
    """
    Ingests data from a given path.

    Args:
        data_source: The path to the data.

    Returns:
        The data as a string.
    """
    try:
        logging.info(f"Reading data from {data_source}")
        data = pd.read_csv(
            data_source,  encoding="unicode_escape", low_memory=False)
        logging.info(f"Data read from {data_source}")
        return data
    except Exception as e:
        logging.error(f"Error reading data from {data_source}: {e}")
        raise e
