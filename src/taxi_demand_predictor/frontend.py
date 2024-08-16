import zipfile
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import streamlit as st
import geopandas as gpd
import pydeck as pdk
from taxi_demand_predictor.paths import DATA_DIR
from taxi_demand_predictor.plots import plot_ts, plot_train_and_target
from taxi_demand_predictor.inference import get_model_predictions, load_batch_of_features_from_store, load_model_from_registry, load_predictions_from_store, get_or_create_feature_view
from taxi_demand_predictor.config import CURRENT_DATE, N_STEPS, DESTINATIONS_TIMES
import asyncio
import tornado.websocket

# Set page configuration
st.set_page_config(layout="wide")

@st.cache_resource
def load_background_data_file():
    URL = 'https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=Shapefile'
    path = DATA_DIR / 'taxi_zones.zip'
    
    # Check if the .shp file already exists
    shp_files = list(DATA_DIR.glob('**/*.shp'))
    if shp_files:
        shp_path = shp_files[0]
        print(f"Shapefile already exists: {shp_path}")
    else:
        response = requests.get(URL)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded file to {path}")
        else:
            raise Exception(f'Error in {URL}: {response.status_code}')
        
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
            print(f"Extracted files to {DATA_DIR}")
        
        shp_files = list(DATA_DIR.glob('**/*.shp'))
        if not shp_files:
            raise FileNotFoundError("No .shp file found in the directory after extraction")
        
        shp_path = shp_files[0]
        print(f"Reading shapefile: {shp_path}")
    
    return gpd.read_file(shp_path).to_crs('EPSG:4326')

@st.cache_data
def pseudocolor(val, minval, maxval, startcolor, stopcolor):
    """
    Convert value in the range minval...maxval to a color in the range
    startcolor to stopcolor. The colors passed and the one returned are
    composed of a sequence of N component values.

    Credits to https://stackoverflow.com/a/10907855
    """
    f = float(val - minval) / (maxval - minval)
    return tuple(f * (b - a) + a for (a, b) in zip(startcolor, stopcolor))

@st.cache_data
def prepare_data(_nyc_map, result):
    df = pd.merge(_nyc_map, result, right_on='pickup_location_id', left_on='objectid', how='inner')
    BLACK, GREENISH = (0, 0, 0), (51, 203, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREENISH))
    return df

def generate_nyc_map(df):
    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=-73.9712,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.05, # Changed from 0.25
        stroked=False,
        filled=True, # Changed
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )

    tooltip = {"html": "<b>Zone:</b> [{objectid}]{zone} <br /> <b>Predicted rides:</b> {predicted_demand}"}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    return r

async def websocket_handler():
    try:
        # Your WebSocket connection and communication logic here
        pass
    except tornado.websocket.WebSocketClosedError:
        print("WebSocket connection closed. Attempting to reconnect...")
        # Implement reconnection logic here
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Run the WebSocket handler
asyncio.run(websocket_handler())

current_date = pd.to_datetime(datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
st.title(f'NYC Taxi Demand Predictor')
st.markdown('<h2 style="font-size:24px;">Predicted Demand For The Next Hour</h2>', unsafe_allow_html=True)
st.markdown(f'<h2 style="font-size:24px;">{current_date}</h2>', unsafe_allow_html=True)

progress_bar = st.sidebar.header('Progress and features:')
progress_bar = st.sidebar.progress(0)

try:
    with st.spinner('Loading data...'):
        nyc_map = load_background_data_file()
        st.sidebar.write('Data loaded successfully')
        progress_bar.progress(1/N_STEPS)
except Exception as e:
    st.sidebar.write(f"Failed to load data: {e}")
    st.error(f"Failed to load data: {e}")
    print(f"Failed to load data: {e}")

try:
    with st.spinner('Fetching batch of inference data...'):
        current_date = CURRENT_DATE
        features_data = load_batch_of_features_from_store(current_date)
        features = features_data.drop(columns=['rides_next_hour'])
        st.sidebar.write('Inference features fetched from the store')
        progress_bar.progress(2/N_STEPS)
        print(f"{features}")
        print(f"{features.shape}")
except Exception as e:
    st.sidebar.write(f"Failed to fetch inference data: {e}")
    st.error(f"Failed to fetch inference data: {e}")
    print(f"Failed to fetch inference data: {e}")

try:
    with st.spinner('Loading model from the registry...'):
        model = load_model_from_registry()
        st.sidebar.write('ML model loaded from the Registry')
        progress_bar.progress(3/N_STEPS)
except Exception as e:
    st.sidebar.write(f"Failed to load model: {e}")
    st.error(f"Failed to load model: {e}")
    print(f"Failed to load model: {e}")

try:
    with st.spinner('Computing model predictions...'):
        result = get_model_predictions(model, features)
        st.sidebar.write('Model predictions computed')
        progress_bar.progress(4/N_STEPS)
except Exception as e:
    st.sidebar.write(f"Failed to compute model predictions: {e}")
    st.error(f"Failed to compute model predictions: {e}")
    print(f"Failed to compute model predictions: {e}")

try:
    with st.spinner(text="Preparing data to plot..."):
        df = prepare_data(nyc_map, result).sample(frac=0.40)
        st.sidebar.write('Plotting data prepared')
        progress_bar.progress(5/N_STEPS)
except Exception as e:
    st.sidebar.write(f"Failed to prepare data for plotting: {e}")
    st.error(f"Failed to prepare data for plotting: {e}")
    print(f"Failed to prepare data for plotting: {e}")

try:
    with st.spinner(text="Generating NYC Map..."):
        r = generate_nyc_map(df)
        st.pydeck_chart(r)
        st.sidebar.write('NYC demand heatmap completed')
        progress_bar.progress(6/N_STEPS)
except Exception as e:
    st.sidebar.write(f"Failed to generate NYC map: {e}")
    st.error(f"Failed to generate NYC map: {e}")
    print(f"Failed to generate NYC map: {e}")

try:
    with st.spinner(text="Plotting time series..."):
        row_index = np.argsort(result['predicted_demand']).values[::-1]
        
        st.markdown(f'<h2 style="font-size:24px;">Top 3 Destination Zones/Times</h2>', unsafe_allow_html=True)
        cols = st.columns(DESTINATIONS_TIMES)
        
        for i in range(DESTINATIONS_TIMES):
            row_id = row_index[i]
            fig = plot_train_and_target(features_data,
                                        sample=row_id,
                                        target_column='rides_next_hour',  # Ensure this is a string
                                        predictions=result['predicted_demand']  # Ensure this is a Series or DataFrame
                                        )
            cols[i].plotly_chart(fig, theme='streamlit', use_container_width=True, width=1000)     
        st.sidebar.write('Time series plot completed')
        progress_bar.progress(7/N_STEPS)
except Exception as e:
    st.sidebar.write(f"Failed to plot time series: {e}")
    st.error(f"Failed to plot time series: {e}")
    print(f"Failed to plot time series: {e}")