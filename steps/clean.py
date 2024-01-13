from zenml import step
import pandas as pd
import logging
from typing import Union


@step
def clean_data(data: pd.DataFrame) -> Union[pd.DataFrame, None]:
    """
    Clean the data by removing duplicates and null values.
    """
    try:
        logging.info("Cleaning data...")
        data.drop_duplicates(keep='last', inplace=True)
        data.dropna(inplace=True)
        # format the date time
        data['date'] = pd.to_datetime(data.date).values

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

        # literacy rate conversion
        data['literacy_rate_perc'] = data.literacy_rate_perc.astype('float32')
        
        logging.info("Data cleaned.")
        return data
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        return None

