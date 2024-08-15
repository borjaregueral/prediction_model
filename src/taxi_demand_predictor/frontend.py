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
from taxi_demand_predictor.inference import get_model_predictions, load_batch_of_features_from_store, load_model_from_registry, load_predictions_from_store


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
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
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
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )

    tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_demand}"}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    return r

current_date = pd.to_datetime(datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
st.title(f'NYC Taxi Demand Predictor')
st.header(f'{current_date}')

progress_bar = st.sidebar.header('Work in progress...')
progress_bar = st.sidebar.progress(0)
N_STEPS = 10

with st.spinner('Loading data...'):
    try:
        nyc_map = load_background_data_file()
        st.sidebar.write('Data loaded successfully')
        progress_bar.progress(1/N_STEPS)
    except Exception as e:
        st.sidebar.write(f"Failed to load data: {e}")
        print(f"Failed to load data: {e}")

with st.spinner('Fetching batch of inference data...'):
    current_date = pd.to_datetime(datetime.utcnow()).floor('h')
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write('inference features fetched from the store')
    progress_bar.progress(2/N_STEPS)
    print(f"{features}")
    print(f"{features.shape}")


with st.spinner('Loading model from the registry...'):
    model = load_model_from_registry()
    st.sidebar.write('ML model was loaded from the Registry')
    progress_bar.progress(3/N_STEPS)


with st.spinner('Computing model predictions...'):
    to_pickup_hour = current_date - timedelta(days=7*10)
    from_pickup_hour = to_pickup_hour - timedelta(days=7*52)
    result = get_model_predictions(model, features)
    st.sidebar.write('Model predictions computed')
    progress_bar.progress(4/N_STEPS)
    

with st.spinner(text="Preparing data to plot..."):
    df = prepare_data(nyc_map, result)
    st.sidebar.write('Data to plot prepared')
    progress_bar.progress(6 / N_STEPS)

with st.spinner(text="Generating NYC Map..."):
    r = generate_nyc_map(df)
    st.pydeck_chart(r)
    st.sidebar.write('NYC demand map completed')
    progress_bar.progress(7 / N_STEPS)