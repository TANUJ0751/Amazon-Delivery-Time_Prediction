
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, cos, sin, asin, sqrt

st.set_page_config(page_title='Amazon Delivery Time Predictor', layout='centered')

@st.cache_resource
def load_model(path='outputs/best_model.joblib'):
    return joblib.load(path)

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

model = load_model()

st.title('Amazon Delivery Time Predictor')
with st.form('input_form'):
    st.subheader('Order & Delivery details')
    agent_age = st.number_input('Agent age', min_value=18, max_value=75, value=28)
    agent_rating = st.slider('Agent rating', min_value=1.0, max_value=5.0, value=4.5, step=0.1)
    store_lat = st.number_input('Store latitude', value=12.95)
    store_lon = st.number_input('Store longitude', value=77.6)
    drop_lat = st.number_input('Drop latitude', value=12.96)
    drop_lon = st.number_input('Drop longitude', value=77.61)
    order_hour = st.number_input('Order hour (0-23)', min_value=0, max_value=23, value=12)
    order_day = st.number_input('Day of week (0=Mon)', min_value=0, max_value=6, value=2)
    pickup_delay = st.number_input('Pickup delay (mins)', min_value=0.0, value=5.0)
    weather = st.selectbox('Weather', ['Clear','Rain','Cloudy','Storm'])
    traffic = st.selectbox('Traffic', ['Low','Medium','High'])
    vehicle = st.selectbox('Vehicle', ['Bike','Car','Van'])
    area = st.selectbox('Area', ['Urban','Metropolitan','Suburban'])
    category = st.selectbox('Category', ['Electronics','Clothing','Grocery','Home'])
    submit = st.form_submit_button('Predict')

if submit:
    dist = haversine(store_lat, store_lon, drop_lat, drop_lon)
    X = pd.DataFrame([{
        'Agent_Age': agent_age,
        'Agent_Rating': agent_rating,
        'distance_km': dist,
        'order_hour': order_hour,
        'order_dayofweek': order_day,
        'pickup_delay_mins': pickup_delay,
        'weather': weather,
        'traffic_level': traffic,
        'Vehicle': vehicle,
        'Area': area,
        'Category': category
    }])
    pred = model.predict(X)[0]
    st.metric('Predicted Delivery Time (hours)', f"{pred:.2f} hrs")
    st.write('Distance (km):', round(dist,2))
