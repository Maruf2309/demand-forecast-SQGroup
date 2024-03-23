import numpy as np
import pandas as pd
import mlflow
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor, XGBRFRegressor
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures, ExpandingWindowFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SmartCorrelatedSelection
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator
import logging, pickle
from typing import Annotated, Tuple, Dict, List, Union
from lightgbm import LGBMRegressor
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import streamlit as st

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
        data = pd.read_parquet(data_source)
        logging.info(f"Data read from {data_source}")
        return data
    except Exception as e:
        logging.error(f"Error reading data from {data_source}: {e}")
        raise e

def clean_data(data: Annotated[pd.DataFrame, 'data']) -> Annotated[pd.DataFrame, 'cleaned_data']:
    """
    Clean the data by removing duplicates, null values, and converting columns to appropriate types.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The cleaned data. None if an error occurs.

    """
    try:
        logging.info("Cleaning data...")
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


def AddLagFeatures(targets: Annotated[pd.DataFrame, 'after added temporal features']) -> Annotated[pd.DataFrame, 'Lag features']:
    """
    Add lag features to the data.
    """
    logging.info(f"Adding lag features to the data.")
    try:
        # Add Lag  Feature
        lagfeatures = LagFeatures(variables=None, periods=[3, 8, 16, 24], freq=None, sort_index=True,
                                  missing_values='raise', drop_original=False)
        lagfeatures.fit(targets[['timestamp', 'net_price', 'qtym']])
        features = lagfeatures.transform(
            targets[['timestamp', 'net_price', 'qtym']])
        # for col in list(features.columns)[3:]:
        #     data[col] = features[col].values
        logging.info(f'==> Successfully processed add_lag_features()')
        return features.drop(['timestamp', 'net_price', 'qtym'], axis=1)
    except Exception as e:
        logging.error(f'in The add_lag_features(): {e}')
        raise e



def AddWindowFeatures(targets: Annotated[pd.DataFrame, 'After lag features added']) -> Annotated[pd.DataFrame, 'window features']:
    """Add window features to the dataframe

    Args:
        data (Union[dd.DataFrame, pd.DataFrame]): The dataframe to add window features to.

    Returns:
        Union[dd.DataFrame, pd.DataFrame]: The dataframe with window features added.
    """
    logging.info("Adding window features to the dataframe")

    try:
        windowfeatures = WindowFeatures(variables=None, window=24, freq=None, sort_index=True,
                                        missing_values='raise', drop_original=False)
        windowfeatures.fit(
            targets[['timestamp', 'net_price', 'qtym']])
        features = windowfeatures.transform(
            targets[['timestamp', 'net_price', 'qtym']])
        # for col in list(features.columns)[3:]:
        #     data[col] = features[col].values
        logging.info(f'==> Successfully processed add_window_features()')
        return features.drop(['timestamp', 'net_price', 'qtym'], axis=1)
    except Exception as e:
        logging.error(f'in add_window_features(): {e}')
        raise e



def AddExpWindowFeatures(targets: Annotated[pd.DataFrame, 'after added temporal features']) -> Annotated[pd.DataFrame, 'added Expanding Window features']:
    """Add Expanding Window Features to the data.
    Args:
        data (pd.DataFrame): The input data.
    Returns:
        pd.DataFrame: The data with added expanding window features.
    """
    try:

        expwindow = ExpandingWindowFeatures(
            variables=None, min_periods=7, functions='std', 
            periods=7, freq=None, sort_index=True, 
            missing_values='raise', drop_original=False
        )
        features = expwindow.fit_transform(targets[['timestamp', 'net_price', 'qtym']])
        
        # # 
        # for col in list(features.columns)[3:]:
        #     data[col] = features[col].values
        return features.drop(['timestamp', 'net_price', 'qtym'], axis=1)
    except Exception as e:
        logging.error(f'in The add_expw_features(): {e}')
        raise e



def merge_all_features(
    targets: Annotated[pd.DataFrame, 'targets'], 
    static_features: Annotated[pd.DataFrame, 'static_features'],
    application_group: Annotated[pd.DataFrame, 'application_group'],
    uses: Annotated[pd.DataFrame, 'uses'],
    mkt: Annotated[pd.DataFrame, 'mkt'],
    
) -> Tuple[Annotated[pd.DataFrame, 'features'], Annotated[pd.Series, 'target'], Annotated[BaseEstimator, 'imputer']]:
    """Merges All Features into One.
    Args:
        data (pd.DataFrame): The input data.
    Returns:
        pd.DataFrame: The data with added expanding window features.
    """
    try:
        logging.info(f'==> merging features...')
        
        # Generate outlet wise timeseries_features
        timeseries_features_outlet_wise = pd.DataFrame()
        for outlet_id in targets.outlet_id.value_counts().index:
            outlet_wise = targets.loc[targets.outlet_id==outlet_id]
            temporal = AddTemporalFeatures(outlet_wise)
            lag_features = AddLagFeatures(outlet_wise)
            window_features = AddWindowFeatures(outlet_wise)
            exp_window_features = AddExpWindowFeatures(outlet_wise)
            outlet_wise_features = pd.concat([outlet_wise[['timestamp','outlet_id',]], temporal, lag_features, window_features, exp_window_features], axis=1)
            timeseries_features_outlet_wise = pd.concat([timeseries_features_outlet_wise, outlet_wise_features], ignore_index=True)
        
        # Merge outlet wise timeseries_features
        targets.merge(timeseries_features_outlet_wise, on=['timestamp','outlet_id',], how='inner')

        
        # Merge application group, uses, mkt
        data = targets.merge(
            application_group, on='outlet_id', how='inner'
        ).merge(
            uses, on='outlet_id', how='inner'
        ).merge(
            mkt, on='outlet_id', how='inner'
        ).merge(
            static_features, on=['timestamp', 'outlet_id'], how='inner'
        ).merge(
            timeseries_features_outlet_wise, on=['timestamp', 'outlet_id'], how='inner')
        
        # Impute Missing Values
        target = data['qtym']
        features = data.drop(columns=['timestamp', 'net_price', 'qtym', 'outlet_id'])
        imputer = KNNImputer(n_neighbors=5).fit(features)
        features = pd.DataFrame(imputer.transform(features), columns=features.columns)
        del data
        return features, target, imputer
    except Exception as e:
        logging.error(f'==> Error when merging features: {e}')
        raise e




# load model
def load_model_by_mlflow_run(uri):
    # Assuming your model is stored in a mlflow run
    # Load model as a PyFuncModel.
    try:
        model_uri = uri
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Create a function to load the data for a specific outlet_id
def load_data_by_outlet(outlet_id, DATA, FEATURES):
    # Assuming your model has a function to load data based on outlet_id
    filtered = DATA.loc[DATA.outlet_id==outlet_id]
    return filtered[FEATURES], filtered[['timestamp', 'target']]


def train_by_outlets(data, target):
    """Train model for each outlet"""
        
    # Train by outlet
    models = {}
    for outlet_id in data.outlet_id.unique():
        model = Prophet(
            growth='linear',
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            yearly_seasonality='auto',
            weekly_seasonality='auto',
            daily_seasonality='auto',
            holidays=None,
            seasonality_mode='additive',
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.8,
            uncertainty_samples=1000,
            stan_backend=None,
            scaling = 'absmax',
            holidays_mode=None,
        )
            
        # Train
        model.fit(
            data[['timestamp', target]].rename(columns={'timestamp': 'ds', target: 'y'}).loc[data.outlet_id==outlet_id]
        )
        models[outlet_id] = model
    # model.fit(data.rename(columns={'timestamp': 'ds', target: 'y'}))
    # models['all'] = model
    return models
            

# Function to save Prophet models for different outlets
def save_models(models_dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(models_dict, f)

# Function to load Prophet models for different outlets
st.cache
def load_models(file_path):
    with open(file_path, 'rb') as f:
        models_dict = pickle.load(f)
    return models_dict


def get_forecast_by_outlet(model, data, outlet_id, target):
    """Get Next 30 day Forecast"""
    return model.predict(data)