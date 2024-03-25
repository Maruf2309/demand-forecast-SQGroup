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
DATA: pd.DataFrame =  pd.read_csv('https://raw.githubusercontent.com/skhapijulhossen/demand-forecast-SQGroup/dev/app/processed.csv', )
MODELS_QTYM: Dict = load_models_from_url(url='https://github.com/skhapijulhossen/demand-forecast-SQGroup/raw/dev/app/qtym_forecasting_models.pkl')
MODELS_NETPRICE: Dict = load_models_from_url(url='https://github.com/skhapijulhossen/demand-forecast-SQGroup/raw/dev/app/net_price_forecasting_models.pkl')
MONTHS:  List[AnyStr] = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January']



# Main Panel
def interfaceApp():
    global MODELS_QTYM, DATA
    try:
        st.title('SQ Group BYA Series Demand')

        # Create a layout with two columns
        panelDemand, panelImportance = st.columns([2,1])

        # Create tabs
        with st.sidebar:
            st.sidebar.title('Target')
            tab_selection = st.radio("________", ("Quantity", "Net Price"))
        


        with panelDemand:
            # Dropdown to select outlet_id
            outlets = sorted(DATA.outlet_id.astype(int).unique().tolist())
            outlet_id = st.selectbox('Select Outlet ID', options=outlets)
            # Tabs
            if tab_selection == 'Quantity':
                actual = DATA[['timestamp', 'qtym']].rename(columns={'timestamp': 'ds', 'qtym': 'y'}).loc[DATA.outlet_id==outlet_id]
                # Plot forecast using Plotly
                future_forecast = forecastPanel(data=actual, model=MODELS_QTYM[outlet_id], outlet=outlet_id)
            else:
                actual = DATA[['timestamp', 'net_price']].rename(columns={'timestamp': 'ds', 'netprice': 'y'}).loc[DATA.outlet_id==outlet_id]
                # Plot forecast using Plotly                
                future_forecast = forecastPanel(data=actual, model=MODELS_QTYM[outlet_id], outlet=outlet_id)
            
        # Numerics
        with panelImportance:
                st.subheader('Forecasted Data')    
                # Display DataFrame
                st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], use_container_width=True, height=600)
            
        # Trends
        if tab_selection == 'Quantity':
            st.subheader('Trend Componenets')
            plott  = plot_components_plotly(MODELS_QTYM[outlet_id], future_forecast)
            st.plotly_chart(plott, use_container_width=True)
        
        if tab_selection == 'Net Price':
            # Trends
            st.subheader('Trend Componenets')
            plott  = plot_components_plotly(MODELS_QTYM[outlet_id], future_forecast)
            st.plotly_chart(plott, use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == '__main__':
    interfaceApp()

