from zenml import step
import pandas as pd
from feature_engine.datetime import DatetimeFeatures
import logging
from typing import Union, Annotated


@step(name='Add Temporal Features', enable_step_logs=True, enable_artifact_metadata=True)
def AddTemporalFeatures(targets: Annotated[pd.DataFrame, 'encoded data']) -> Annotated[pd.DataFrame, 'temporal features']:
    features_to_extract = [
        "month", "quarter", "semester", "week", "day_of_week", "day_of_month",
        "day_of_year", "weekend", "month_start", "month_end", "quarter_start",
        "quarter_end", "year_start", "year_end"
    ]

    try:
        logging.info(f'==> Processing AddTemporalFeatures()')
        temporal = DatetimeFeatures(
            features_to_extract=features_to_extract).fit_transform(targets[['timestamp']])
        # for col in temporal.columns:
        #     data.loc[:, col] = temporal[col].values
        logging.info(f'==> Successfully processed AddTemporalFeatures()')
        return temporal
    except Exception as e:
        logging.error(f'==> Error in AddTemporalFeatures()')
        raise e
