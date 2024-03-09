from zenml import step
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from typing import Union
import logging
import numpy as np
from typing_extensions import Annotated, Tuple


@step(name='Encode and Aggregate Features', enable_artifact_metadata=True, enable_step_logs=True)
def encode_and_aggregate_data(
    data: Annotated[pd.DataFrame, 'cleaned_data']
) -> Tuple[Annotated[pd.DataFrame, 'target'], Annotated[pd.DataFrame, 'static features'], Annotated[pd.DataFrame, 'aggregated application_group table'], Annotated[pd.DataFrame, 'aggregated uses tabe'], Annotated[pd.DataFrame, 'aggregated mkt table']]:
    """
    Encode categorical features of the data.

    Args:
        data: Dataframe containing the categorical features.

    Returns:
        Dataframe containing the encoded categorical features OR None.

    """
    try:
        logging.info('Encoding categorical features...')
        
        # HASH FEATURES category
        data['category'] = data.category.apply(
            lambda cat: {'Domestic': 1, 'Power': 0}[cat])
        data['grade'] = data.grade.apply(
            lambda cat: {'Grade1': 1, 'Grade2': 2,
                         'Grade3': 3, 'Grade4': 4}[cat]
        )
        data['ecoind'] = data.ecoind.apply(
            lambda cat: {'Medium': 2, 'High': 4, 'Low': 2, 'Poor': 1}[cat]
        )

        # OneHot Encoding
        data = pd.get_dummies(data, columns=['division', 'region'])
        
        # renaming cols
        data.columns = [col.lower().strip().replace(' ', '_')
                        for col in data.columns]
        
        for column in data.select_dtypes('bool').columns:
            data[column] = data[column].astype(int)
    

        # optimize for memory
        for col in data.select_dtypes('int64').columns:
            data[col] = data[col].astype('int32')

        # Aggregate targets by outlet_id
        targets = data.pivot_table(index=['timestamp','outlet_id'], aggfunc={
                'net_price': 'mean',
                'qtym': 'mean',
            }
        ).reset_index()
        
        # Aggreate static feature by outlet_id
        static_features = data[['timestamp','outlet_id','wire', 'rm',
           'fy', 'grade', 'noc',
           'dfc', 'area_km2', 'population', 'literacy_rate_perc', 'pcx', 'excnts',
           'exach', 'trc', 'tlcolt', 'tmtm', 'ecoind', 'sf', 'sop', 'pminx',
           'tms_cr', 'mas', 'kpi', ]].groupby(by=['timestamp','outlet_id'],).mean().reset_index()
        
        # aggreated appliatin group by outlet_id
        application_group = pd.DataFrame(columns=['General', 'Moderate', 'Rich', 'Industry'])
        for outlet in data.outlet_id.value_counts().index:
            ratio = data.loc[data.outlet_id==outlet, 'application_group'].value_counts(normalize=True).to_dict()
            application_group.loc[outlet] = ratio
        application_group = application_group.fillna(0).reset_index().rename(columns={'index':'outlet_id'}).astype(np.float32)
        
        # Aggregated uses by outlet_id
        uses = pd.DataFrame(columns=[
            'House Wiring', 'Fan & Lighting Connection',
            'Air Condition & Washing Machine, Heavy Item', 'Lift & Heavy Item',
            'Earthing', 'Industry, Machineries'
            ]
        )
        for outlet in data.outlet_id.value_counts().index:
            ratio = data.loc[data.outlet_id==outlet, 'uses'].value_counts(normalize=True).to_dict()
            uses.loc[outlet] = ratio
        uses = uses.fillna(0).reset_index().rename(columns={'index':'outlet_id'}).astype(np.float32)
        
        # Aggregated mkt ratio by outlet_id
        mkt = pd.DataFrame(columns=
            ['Urban', 'Rural', 'Semi Urban', 'Others']
        )
        for outlet in data.outlet_id.value_counts().index:
            ratio = data.loc[data.outlet_id==outlet, 'mkt'].value_counts(normalize=True).to_dict()
            mkt.loc[outlet] = ratio
        mkt = mkt.fillna(0).reset_index().rename(columns={'index':'outlet_id'}).astype(np.float32)
        logging.info('Encoding categorical features completed.')
        return targets, static_features, application_group, uses, mkt
    except Exception as e:
        logging.error(f'Error encoding categorical features: {e}')
        return None