from zenml import step
import logging
import pandas as pd
from typing import Union, Annotated
from sklearn.preprocessing import StandardScaler
import joblib
import os
import config

logger = logging.getLogger(__name__)


@step(name='Data Scaling', enable_step_logs=True, enable_artifact_metadata=True)
def scale_data(data: Annotated[pd.DataFrame, 'data to scale'],
               temporalFeatures: Annotated[pd.DataFrame, 'temporal features'],
               lagFeatures: Annotated[pd.DataFrame, 'lag features'],
               windowFeatures: Annotated[pd.DataFrame, 'window features'],
               expWindowFeatures: Annotated[pd.DataFrame,
                                            'Expanding window features']
               ) -> Annotated[pd.DataFrame, 'standardized data']:
    """Scaling step.
    Args:
        data: Input data.
    Returns:
        Normalized data.
    """
    try:
        logger.info(f'==> Processing scale_data()')
        scaler = StandardScaler()
        # Assuming the data is a pandas DataFrame
        temp = data[['timestamp', 'net_price', 'qtym']]
        data = pd.concat([temporalFeatures, lagFeatures,
                         windowFeatures, expWindowFeatures], axis=1)
        scaler.fit(data)
        data = pd.concat(
            [temp, pd.DataFrame(scaler.transform(data), columns=data.columns)], axis=1)
        del temp
        # save Scaler model
        joblib.dump(scaler, os.path.join(config.ARTIFACTS_DIR, 'scaler.pkl'))
        logger.info(
            f'Scaler model saved to {os.path.join(config.ARTIFACTS_DIR, "scaler.pkl")}')
        data.dropna(inplace=True)
        print(data.head())
        data.to_parquet(config.FEATURE_STORE, index=False)
        return data
    except Exception as e:
        logger.error(f"in scale_data(): {e}")
        raise e


if __name__ == "__main__":
    data = pd.read_csv("data/train.csv")
    print(scale_data(data))
