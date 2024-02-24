from zenml import step
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from typing import Union
import logging
import numpy as np
from typing_extensions import Annotated


@step(enable_cache=True)
def encode_data(data: Annotated[pd.DataFrame, 'cleaned_data']) -> Annotated[pd.DataFrame, 'encoded_data']:
    """
    Encode categorical features of the data.

    Args:
        data: Dataframe containing the categorical features.

    Returns:
        Dataframe containing the encoded categorical features OR None.

    """
    try:
        logging.info('Encoding categorical features...')
        data['category'] = data.category.apply(
            lambda cat: {'Domestic': 1, 'Power': 0}[cat])

        data['grade'] = data.grade.apply(
            lambda cat: {'Grade1': 1, 'Grade2': 2,
                         'Grade3': 3, 'Grade4': 4}[cat]
        )

        data['uses'] = data.uses.apply(
            lambda cat: {
                'House Wiring': 1,
                'Fan & Lighting Connection': 2,
                'Air Condition & Washing Machine, Heavy Item': 3,
                'Lift & Heavy Item': 4,
                'Earthing': 5,
                'Industry, Machineries': 6
            }[cat]
        )

        data['application_group'] = data.application_group.apply(
            lambda cat: {
                'General': 1, 'Moderate': 2, 'Rich': 3, 'Industry': 4
            }[cat]
        )

        data['ecoind'] = data.ecoind.apply(
            lambda cat: {'Medium': 2, 'High': 4, 'Low': 2, 'Poor': 1}[cat]
        )

        data['division'] = data.division.apply(
            lambda cat: {
                'Dhaka': 1,
                'Chittagong': 2,
                'Khulna': 3,
                'Rajshahi': 4,
                'Mymensingh': 5,
                'Sylhet': 6,
                'Rangpur': 7,
                'Barishal': 8}[cat]
        )

        data['mkt'] = data.mkt.str.strip(' ').apply(
            lambda cat: {
                'Urban': 4,
                'Semi Urban': 3,
                'Rural': 2,
                'Others': 1,
            }[cat]
        )

        # base_size ordinal encoding
        encoder = OrdinalEncoder(
            categories='auto', encoded_missing_value=-1,
            handle_unknown='error'
        )
        encoder.fit(data[['base_size']], y=data.net_price)
        data['base_size'] = encoder.transform(data[['base_size']])

        # optimize for memory
        for col in data.select_dtypes('int64').columns:
            data[col] = data[col].astype('int32')

        # Aggregate
        data = data[['timestamp', 'net_price', 'qtym']]
        data = data.groupby(by='timestamp').mean().reset_index()

        logging.info('Encoding categorical features completed.')
        return data
    except Exception as e:
        logging.error(f'Error encoding categorical features: {e}')
        return None
