import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from typing  import List, Tuple, Dict, Set, Any, Annotated, AnyStr
from datetime import timedelta
from prophet import Prophet
import pickle
from model import *
from utils import *
from components import forecastPanel
from streamlit_option_menu import option_menu
import plotly.graph_objs as go


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

# Globals
# MODEL_URI: str = 'runs:/ae501105c42f4987b526ea740568507a/LGB-QTYM-0.7456832147065215'
# MODEL: mlflow.pyfunc.PyFuncModel = load_model_by_mlflow_run(MODEL_URI)
OUTLET_DATA: pd.DataFrame =  pd.read_parquet('https://raw.githubusercontent.com/skhapijulhossen/demand-forecast-SQGroup/dev/app/processed.parquet', )
REGION_DATA: pd.DataFrame =  pd.read_parquet('https://raw.githubusercontent.com/skhapijulhossen/demand-forecast-SQGroup/dev/app/region_aggreagted.parquet', )
DIVISION_DATA: pd.DataFrame =  pd.read_parquet('https://raw.githubusercontent.com/skhapijulhossen/demand-forecast-SQGroup/dev/app/division_aggreagted.parquet', )
MODELS_OUTLET_QTYM: Dict = load_models_from_url(url='https://github.com/skhapijulhossen/demand-forecast-SQGroup/raw/dev/app/qtym_forecasting_models.pkl')
MODELS_OUTLET_NETPRICE: Dict = load_models_from_url(url='https://github.com/skhapijulhossen/demand-forecast-SQGroup/raw/dev/app/net_price_forecasting_models.pkl')
MODELS_REGION_QTYM: Dict = load_models_from_url(url='https://github.com/skhapijulhossen/demand-forecast-SQGroup/raw/dev/app/qtym_forecasting_models_by_region.pkl')
MODELS_REGION_NETPRICE: Dict = load_models_from_url(url='https://github.com/skhapijulhossen/demand-forecast-SQGroup/raw/dev/app/net_price_forecasting_models_by_region.pkl')
MODELS_DIVISION_QTYM: Dict = load_models_from_url(url='https://github.com/skhapijulhossen/demand-forecast-SQGroup/raw/dev/app/qtym_forecasting_models_by_division.pkl')
MODELS_DIVISION_NETPRICE: Dict = load_models_from_url(url='https://github.com/skhapijulhossen/demand-forecast-SQGroup/raw/dev/app/net_price_forecasting_models_by_division.pkl')
MONTHS:  List[AnyStr] = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January']



# Main Panel
def interfaceApp():
    global MODELS_OUTLET_QTYM, MODELS_OUTLET_NETPRICE, MODELS_DIVISION_NETPRICE, MODELS_DIVISION_QTYM, MODELS_REGION_NETPRICE, MODELS_REGION_QTYM
    global OUTLET_DATA, REGION_DATA, DIVISION_DATA

    try:
        st.title('SQ Group BYA Series Demand')

        # Create a layout with two columns
        panelDemand, panelData= st.columns([2,1])
        
        with st.sidebar:
            selected = option_menu(
                "SQ Analytics", ['Outlets', 'Regions', 'Divisions'], 
                icons=['house', 'gear'], menu_icon="cast", default_index=0
                )


        with panelDemand:
            # Create tabs
            tab_selection = st.radio("Forecast", ("Quantity", "Net Price"))


            if selected == 'Outlets':
                # Dropdown to select outlet_id
                outlets = sorted(OUTLET_DATA.outlet_id.astype(int).unique().tolist())
                outlet_id = st.selectbox('Select Outlet ID', options=outlets)
                # Tabs
                if tab_selection == 'Quantity':
                    actual = OUTLET_DATA[['timestamp', 'qtym']].rename(columns={'timestamp': 'ds', 'qtym': 'y'}).loc[OUTLET_DATA.outlet_id==outlet_id]
                    # Plot forecast using Plotly
                    future_forecast = forecastPanel(data=actual, model=MODELS_OUTLET_QTYM[outlet_id], aggregated_by=outlet_id)
                else:
                    actual = OUTLET_DATA[['timestamp', 'net_price']].rename(columns={'timestamp': 'ds', 'net_price': 'y'}).loc[OUTLET_DATA.outlet_id==outlet_id]
                    # Plot forecast using Plotly                
                    future_forecast = forecastPanel(data=actual, model=MODELS_OUTLET_NETPRICE[outlet_id], aggregated_by=outlet_id)
                
                # Trends
                if tab_selection == 'Quantity':
                    st.subheader('Trend & Seasonailty')
                    plott  = plot_components_plotly(MODELS_OUTLET_QTYM[outlet_id], future_forecast)
                    st.plotly_chart(plott, use_container_width=True)
                
                if tab_selection == 'Net Price':
                    # Trends
                    st.subheader('Trend & Seasonailty')
                    plott  = plot_components_plotly(MODELS_OUTLET_NETPRICE[outlet_id], future_forecast)
                    st.plotly_chart(plott, use_container_width=True)
            elif selected == 'Regions':
                # Dropdown to select region
                regions = sorted(REGION_DATA.region.unique().tolist())
                region = st.selectbox('Select Region', options=regions)
                # Tabs
                if tab_selection == 'Quantity':
                    actual = REGION_DATA[['timestamp', 'qtym']].rename(columns={'timestamp': 'ds', 'qtym': 'y'}).loc[REGION_DATA.region==region]
                    # Plot forecast using Plotly
                    future_forecast = forecastPanel(data=actual, model=MODELS_REGION_QTYM[region], aggregated_by=region)
                else:
                    actual = REGION_DATA[['timestamp', 'net_price']].rename(columns={'timestamp': 'ds', 'net_price': 'y'}).loc[REGION_DATA.region==region]
                    # Plot forecast using Plotly                
                    future_forecast = forecastPanel(data=actual, model=MODELS_REGION_NETPRICE[region], aggregated_by=region)
                
                # Trends
                if tab_selection == 'Quantity':
                    st.subheader('Trend & Seasonailty')
                    plott  = plot_components_plotly(MODELS_REGION_QTYM[region], future_forecast)
                    st.plotly_chart(plott, use_container_width=True)
                
                if tab_selection == 'Net Price':
                    # Trends
                    st.subheader('Trend & Seasonailty')
                    plott  = plot_components_plotly(MODELS_REGION_NETPRICE[region], future_forecast)
                    st.plotly_chart(plott, use_container_width=True)
            else:
                # Dropdown to select region
                divisions = sorted(DIVISION_DATA.division.unique().tolist())
                division = st.selectbox('Select Division', options=divisions)
                # Tabs
                if tab_selection == 'Quantity':
                    actual = DIVISION_DATA[['timestamp', 'qtym']].rename(columns={'timestamp': 'ds', 'qtym': 'y'}).loc[DIVISION_DATA.division==division]
                    # Plot forecast using Plotly
                    future_forecast = forecastPanel(data=actual, model=MODELS_DIVISION_QTYM[division], aggregated_by=division)
                else:
                    actual = DIVISION_DATA[['timestamp', 'net_price']].rename(columns={'timestamp': 'ds', 'net_price': 'y'}).loc[DIVISION_DATA.division==division]
                    # Plot forecast using Plotly                
                    future_forecast = forecastPanel(data=actual, model=MODELS_DIVISION_NETPRICE[division], aggregated_by=division)
                
                # Trends
                if tab_selection == 'Quantity':
                    st.subheader('Trend & Seasonailty')
                    plott  = plot_components_plotly(MODELS_DIVISION_QTYM[division], future_forecast)
                    st.plotly_chart(plott, use_container_width=True)
                
                if tab_selection == 'Net Price':
                    # Trends
                    st.subheader('Trend & Seasonailty')
                    plott  = plot_components_plotly(MODELS_DIVISION_NETPRICE[division], future_forecast)
                    st.plotly_chart(plott, use_container_width=True)
            
            # Numerics
            with panelData:
                    # st.subheader('Forecasted Data')    
                    # Display DataFrame
                    hist = px.histogram(data_frame=future_forecast[['ds', 'yhat']], y='ds', x='yhat')
                    # st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], use_container_width=True, height=700)
                    st.plotly_chart(figure_or_data=hist, use_container_width=True)


    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == '__main__':
    interfaceApp()

