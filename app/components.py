import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Tuple
from prophet.plot import plot_plotly
import plotly.express as px
from prophet import Prophet



def forecastPanel(data: pd.DataFrame, model: Prophet, aggregated_by: Any) -> pd.DataFrame:
    """
    Function to display the forecast panel
    """
    try:                
        st.subheader(f'Demand Forecast for {aggregated_by}')
        # Recursive forecast for the upcoming month
        future = model.make_future_dataframe(periods=90)
        future_forecast = model.predict(future)
        ffig = plot_plotly(model, future_forecast, trend=True)
        st.plotly_chart(ffig, use_container_width=True)
        return future_forecast
    except Exception as e:
        st.error(f'Error: {e}')
        st.stop()
        
