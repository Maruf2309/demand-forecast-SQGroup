from zenml import step
import pandas as pd
import logging
from typing import Union
from typing_extensions import Annotated


@step(name='Clean Data', enable_cache=False, enable_step_logs=True)
def clean_data(data: Annotated[pd.DataFrame, 'data']) -> Annotated[pd.DataFrame, 'cleaned_data']:
    """
    Clean the data by removing duplicates, null values, and converting columns to appropriate types.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The cleaned data. None if an error occurs.

    """
    try:
        data.drop_duplicates(keep='last', inplace=True)
        data.dropna(inplace=True)
        data.drop(columns=['client_id', 'CID', 'Base Size'], inplace=True)
        
        # format the date time
        data['date'] = pd.to_datetime(data.date).values

        # Sort
        data.sort_values(by='date', inplace=True)

        # renaming cols
        data.columns = [col.lower().strip().replace(' ', '_')
                        for col in data.columns]
        data.rename(
            {'area_(km)^2': 'area_km2', 'population_(approx.)': 'population',
             'literacy_rate_(%)': 'literacy_rate_perc'},
            axis=1, inplace=True)

        # optimizing for memory
        for col in data.select_dtypes('float64').columns:
            data[col] = data[col].astype('float32')

        for col in data.select_dtypes('int64').columns:
            data[col] = data[col].astype('int32')

        # lType conversion
        data['literacy_rate_perc'] = data.literacy_rate_perc.astype('float32')
        data['kpi'] = data.kpi.astype('float16')
        data['tmtm'] = data.tmtm.astype('float32')

        # rename date -> timestamp
        data.rename({'date': 'timestamp'}, axis=1, inplace=True)

        logging.info("Data cleaned.")
        return data
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise e