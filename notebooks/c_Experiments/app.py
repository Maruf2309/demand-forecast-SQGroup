import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import plotly.express as px
from typing  import List, Tuple, Dict, Set, Any, Annotated, AnyStr
from utils import *
from datetime import timedelta

# Globals
FEATURES = [
    'net_price_lag_16', 'Earthing', 'Urban', 'Others', 'grade', 'pminx', 'excnts', 'Fan & Lighting Connection', 
    'kpi', 'sop', 'Rural', 'exach', 'qtym_window_24_mean', 'Rich', 'tmtm', 'Semi Urban', 'tlcolt', 'timestamp_day_of_month', 
    'sf', 'mas', 'noc', 'timestamp_day_of_week', 'fy', 'ecoind', 'tms_cr', 'literacy_rate_perc', 'area_km2', 'qtym_expanding_std', 
    'qtym_lag_24', 'pcx', 'Industry', 'dfc', 'House Wiring', 'wire', 'net_price_lag_3', 'Industry, Machineries', 'qtym_lag_8', 'trc'
    ]
# MODEL_URI: str = 'runs:/ae501105c42f4987b526ea740568507a/LGB-QTYM-0.7456832147065215'
# MODEL: mlflow.pyfunc.PyFuncModel = load_model_by_mlflow_run(MODEL_URI)
DATA: pd.DataFrame =  pd.read_parquet('processed.parquet')
MODELS_QTYM: Dict = load_models(file_path='qtym_forecasting_models.pkl')
MONTHS:  List[AnyStr] = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January']


def interfaceApp():
    global MODELS_QTYM, DATA
    try:
        # Set main panel
        st.set_page_config(
            page_title="SQ Analytics",
            page_icon="🧊",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://www.extremelycoolapp.com/help',
                'Report a bug': "https://www.extremelycoolapp.com/bug",
                'About': "# This is a header. This is an *extremely* cool app!"
            }
        )
        st.title('SQ Group BYA Series Demand')

        # Create a layout with two columns
        col1, col2 = st.columns([2, 1])

        # Sidebar for selectors
        st.sidebar.title('Fiilters')
        panelDemand, panelImportance = st.columns([2,1])

        # Dropdown to select outlet_id
        outlets = DATA.outlet_id.astype(int).unique().tolist()
        outlet_id = st.sidebar.selectbox('Select Outlet ID', options=outlets)

        target = 'qtym'
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
        selected_year = 2023; selected_month = 'January'
        actual = DATA[['timestamp', target]].rename(columns={'timestamp': 'ds', target: 'y'}).loc[DATA.outlet_id==outlet_id]
    
        forecast = MODELS_QTYM[outlet_id].predict(actual)
        # Filter data by year
        years:List = sorted(actual['ds'].dt.year.unique(), reverse=True)
        selected_year = st.sidebar.selectbox('Select Year', options=years)
        if selected_year:
            actual = DATA[['timestamp', target]].rename(columns={'timestamp': 'ds', target: 'y'}).loc[
                ((DATA.outlet_id==outlet_id) & (DATA.timestamp.dt.year==selected_year))
            ]
            forecast = MODELS_QTYM[outlet_id].predict(actual)


            # Filter data by month
            months = sorted(actual['ds'].dt.month_name().unique())
            selected_month = st.sidebar.selectbox('Select Month', options=months)
            if selected_month:
                actual = DATA[['timestamp', target]].rename(columns={'timestamp': 'ds', target: 'y'}).loc[
                    ((DATA.outlet_id==outlet_id) & (DATA.timestamp.dt.year==selected_year) & (DATA.timestamp.dt.month_name()==selected_month))
                ]
                forecast = MODELS_QTYM[outlet_id].predict(actual)

        
        # Plot forecast using Plotly
        with panelDemand:
            # Recursive forecast for the upcoming month
            if st.sidebar.button('Generate Forecast for Next Month'):
                for_month = MONTHS[MONTHS.index(selected_month) + 1]
                future = MODELS_QTYM[outlet_id].make_future_dataframe(periods=30)
                future_forecast = MODELS_QTYM[outlet_id].predict(future)
                future_forecast = future_forecast.loc[(future_forecast.ds.dt.year==selected_year) & (future_forecast.ds.dt.month_name()==for_month)]

                ffig = px.line()
                ffig.add_scatter(x=future_forecast['ds'], y=future_forecast['trend'], mode='lines', name='Forecasted Trend')
                ffig.add_scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='Forecasted Demand')
                ffig.add_scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'], mode='lines', name='Forecasted Min Demand')
                ffig.add_scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'], mode='lines', name='Forecasted Max Demand')
                st.plotly_chart(ffig, use_container_width=True)
            else:
                # Plot actual demand and forecast using Plotly
                st.subheader(f'Demand Forecast for Outlet {outlet_id} - {selected_month} {selected_year}')
                fig = px.line()
                fig.add_scatter(x=actual['ds'], y=actual['y'], mode='lines', name='Actual Demand')
                fig.add_scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Forecasted Trend')
                fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Demand')
                fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Forecasted Min Demand')
                fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Forecasted Max Demand')
                # fig.update_layout(title=f'Demand Forecast for Outlet {outlet_id} - {selected_month} {selected_year}')
                # fig = plot_plotly(MODELS_QTYM[outlet_id], forecast)
                st.plotly_chart(fig, use_container_width=True)


        # # Plot feature importances using Plotly
        # st.subheader('Feature Importances')
        with panelImportance:
                st.subheader('Trend')    
                # Display DataFrame
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
        
        # Trends
        st.subheader('Trend Componenets')
        plott  = plot_components_plotly(MODELS_QTYM[outlet_id], forecast)
        st.plotly_chart(plott, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == '__main__':
    interfaceApp()

