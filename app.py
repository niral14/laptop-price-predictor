import streamlit as st
import pickle
import numpy as np
import gzip

# Load the trained model and dataset correctly using gzip
with gzip.open('pipe.pkl.gz', 'rb') as f:
    pipe = pickle.load(f)

with gzip.open('df.pkl.gz', 'rb') as f:
    df = pickle.load(f)

st.title("Laptop Price Predictor")

# Function to pre-fill inputs based on button click
def set_configuration(config):
    st.session_state['company'] = config['company']
    st.session_state['type'] = config['type']
    st.session_state['ram'] = config['ram']
    st.session_state['weight'] = config['weight']
    st.session_state['touchscreen'] = config['touchscreen']
    st.session_state['ips'] = config['ips']
    st.session_state['screen_size'] = config['screen_size']
    st.session_state['resolution'] = config['resolution']
    st.session_state['cpu'] = config['cpu']
    st.session_state['hdd'] = config['hdd']
    st.session_state['ssd'] = config['ssd']
    st.session_state['gpu'] = config['gpu']
    st.session_state['os'] = config['os']

# Default values (predefined laptop configurations)
configurations = {
    "Windows i3": {
        'company': 'Dell',
        'type': 'Notebook',
        'ram': 8,
        'weight': 1.6,
        'touchscreen': 'No',
        'ips': 'Yes',
        'screen_size': 14.0,
        'resolution': '1920x1080',
        'cpu': 'Intel Core i3',
        'hdd': 0,
        'ssd': 256,
        'gpu': 'Intel',
        'os': 'Windows'
    },
    "Windows i5": {
        'company': 'HP',
        'type': 'Ultrabook',
        'ram': 8,
        'weight': 1.5,
        'touchscreen': 'No',
        'ips': 'Yes',
        'screen_size': 15.6,
        'resolution': '1920x1080',
        'cpu': 'Intel Core i5',
        'hdd': 0,
        'ssd': 512,
        'gpu': 'Intel',
        'os': 'Windows'
    },
    "Windows i7": {
        'company': 'Lenovo',
        'type': 'Ultrabook',
        'ram': 16,
        'weight': 1.8,
        'touchscreen': 'No',
        'ips': 'Yes',
        'screen_size': 15.6,
        'resolution': '2560x1440',
        'cpu': 'Intel Core i7',
        'hdd': 0,
        'ssd': 1024,
        'gpu': 'NVIDIA',
        'os': 'Windows'
    },
    "MacBook Air": {
        'company': 'Apple',
        'type': 'Ultrabook',
        'ram': 8,
        'weight': 1.24,
        'touchscreen': 'No',
        'ips': 'Yes',
        'screen_size': 13.3,
        'resolution': '2560x1600',
        'cpu': 'Apple M1',
        'hdd': 0,
        'ssd': 256,
        'gpu': 'Apple',
        'os': 'Mac'
    }
}

# Buttons to set configurations
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Windows i3"):
        set_configuration(configurations["Windows i3"])
with col2:
    if st.button("Windows i5"):
        set_configuration(configurations["Windows i5"])
with col3:
    if st.button("Windows i7"):
        set_configuration(configurations["Windows i7"])
with col4:
    if st.button("MacBook Air"):
        set_configuration(configurations["MacBook Air"])

# User Inputs (Pre-filled if button clicked)
company = st.selectbox('Brand', df['Company'].unique(), key='company')
type = st.selectbox('Type', df['TypeName'].unique(), key='type')
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], key='ram')
weight = st.number_input('Weight of the Laptop', key='weight')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'], key='touchscreen')
ips = st.selectbox('IPS', ['No', 'Yes'], key='ips')
screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.0, key='screen_size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900',
                                                 '3840x2160', '3200x1800', '2880x1800',
                                                 '2560x1600', '2560x1440', '2304x1440'], key='resolution')
cpu = st.selectbox('CPU', df['Cpu brand'].unique(), key='cpu')
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048], key='hdd')
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024], key='ssd')
gpu = st.selectbox('GPU', df['Gpu brand'].unique(), key='gpu')
os = st.selectbox('OS', df['os'].unique(), key='os')

# Predict Price Button
if st.button('Predict Price'):
    # Convert touchscreen and IPS to numerical format
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Calculate PPI (Pixels Per Inch)
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Prepare input array
    query = np.array([company, type, ram, weight, touchscreen, ips,
                      ppi, cpu, hdd, ssd, gpu, os], dtype=object).reshape(1, -1)

    # Predict price
    predicted_price = np.exp(pipe.predict(query)[0])

    st.title(f"The predicted price of this configuration is ₹{int(predicted_price):,}")
