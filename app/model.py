import pickle
import requests
from io import BytesIO
import streamlit as st
from prophet import Prophet


@st.cache
def train_by(data, target, aggregate_by='outlet_id'):
    """Train model for each outlet"""
        
    # Train by outlet
    models = {}
    for key in data['outlet_id'].unique():
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
            data[['timestamp', target]].rename(columns={'timestamp': 'ds', target: 'y'}).loc[data['outlet_id']==key]
        )
        models[key] = model
    # model.fit(data.rename(columns={'timestamp': 'ds', target: 'y'}))
    # models['all'] = model
    return models










        # Assuming your model has a function to get forecasted demand
        # forecast = MODELS_QTYM[1001].predict(
        #     DATA[['timestamp', target]].rename(columns={'timestamp': 'ds', target: 'y'}).loc[DATA.outlet_id==outlet_id][['ds']]
        # )
        
        # # Get Feature Importances
        # feature_importances = LGBMRegressor().fit(features.loc[forecast.index].values, forecast['demand']).feature_importances_
        # feature_importances: pd.DataFrame = pd.DataFrame(
        #     {
        #         'feature': features.columns,
        #         'importance': feature_importances/feature_importances.max()
        #     }
        # )
        # feature_importances.sort_values(by='importance', ascending=False, inplace=True)
        # init
        # selected_year = st.sidebar.selectbox('Select Year', options=years)
        # if selected_year:
        #     actual = DATA[['timestamp', target]].rename(columns={'timestamp': 'ds', target: 'y'}).loc[
        #         ((DATA.outlet_id==outlet_id) & (DATA.timestamp.dt.year==selected_year))
        #     ]
        #     forecast = MODELS_QTYM[outlet_id].predict(actual)


        #     # Filter data by month
        #     months = sorted(actual['ds'].dt.month_name().unique())
        #     selected_month = st.sidebar.selectbox('Select Month', options=months)
        #     if selected_month:
        #         actual = DATA[['timestamp', target]].rename(columns={'timestamp': 'ds', target: 'y'}).loc[
        #             ((DATA.outlet_id==outlet_id) & (DATA.timestamp.dt.year==selected_year) & (DATA.timestamp.dt.month_name()==selected_month))
        #         ]
        #         forecast = MODELS_QTYM[outlet_id].predict(actual)


@st.cache_data
def load_models_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        models_dict = pickle.load(BytesIO(response.content))
        return models_dict
    else:
        raise Exception("Failed to fetch models from URL")