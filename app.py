import streamlit as st
import pickle
import numpy as np
import gzip

# Load trained model and dataset
with gzip.open('pipe.pkl.gz', 'rb') as f:
    pipe = pickle.load(f)

with gzip.open('df.pkl.gz', 'rb') as f:
    df = pickle.load(f)

st.title("Laptop Price Predictor")

# Fixing CPU Brand Error
if 'Cpu brand' in df.columns:
    df = df.dropna(subset=['Cpu brand'])  # Remove NaN values
    df['Cpu brand'] = df['Cpu brand'].astype(str).str.strip()
    cpu_options = sorted(df['Cpu brand'].unique().tolist())  # Sorted for consistency

    # Ensure Apple M1 is included
    if "Apple M1" not in cpu_options:
        cpu_options.append("Apple M1")
else:
    st.error("Error: 'Cpu brand' column not found in DataFrame.")
    cpu_options = ["Unknown"]

st.write("Available CPU Brands:", cpu_options)  # Debugging info

# Laptop Configurations (HDD values checked below)
configurations = {
    "Windows i3": {'company': 'Dell', 'type': 'Notebook', 'ram': 8, 'weight': 1.6,
                   'touchscreen': 'No', 'ips': 'Yes', 'screen_size': 14.0,
                   'resolution': '1920x1080', 'cpu': 'Intel Core i3', 'hdd': 0,
                   'ssd': 256, 'gpu': 'Intel', 'os': 'Windows'},
    
    "Windows i5": {'company': 'HP', 'type': 'Ultrabook', 'ram': 8, 'weight': 1.5,
                   'touchscreen': 'No', 'ips': 'Yes', 'screen_size': 15.6,
                   'resolution': '1920x1080', 'cpu': 'Intel Core i5', 'hdd': 0,
                   'ssd': 512, 'gpu': 'Intel', 'os': 'Windows'},
    
    "Windows i7": {'company': 'Lenovo', 'type': 'Ultrabook', 'ram': 16, 'weight': 1.8,
                   'touchscreen': 'No', 'ips': 'Yes', 'screen_size': 15.6,
                   'resolution': '2560x1440', 'cpu': 'Intel Core i7', 'hdd': 0,
                   'ssd': 1024, 'gpu': 'NVIDIA', 'os': 'Windows'},
    
    "MacBook Air": {'company': 'Apple', 'type': 'Ultrabook', 'ram': 8, 'weight': 1.24,
                    'touchscreen': 'No', 'ips': 'Yes', 'screen_size': 13.3,
                    'resolution': '2560x1600', 'cpu': 'Apple M1', 'hdd': 0,
                    'ssd': 256, 'gpu': 'Apple', 'os': 'Mac'}
}

# Pre-fill inputs based on button click
def set_configuration(config):
    for key, value in config.items():
        st.session_state[key] = value

# Buttons for predefined configurations
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

# User Inputs
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
cpu = st.selectbox('CPU', cpu_options, key='cpu')  # Fixed issue here
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048], key='hdd')
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024], key='ssd')
gpu = st.selectbox('GPU', df['Gpu brand'].unique(), key='gpu')
os = st.selectbox('OS', df['os'].unique(), key='os')

# Predict Price Button
if st.button('Predict Price'):
    try:
        # Convert categorical values
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        # Calculate PPI
        X_res, Y_res = map(int, resolution.split('x'))
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        # Prepare input for prediction
        query = np.array([company, type, ram, weight, touchscreen, ips,
                          ppi, cpu, hdd, ssd, gpu, os], dtype=object).reshape(1, -1)

        # Predict price
        predicted_price = np.exp(pipe.predict(query)[0])

        st.title(f"The predicted price of this configuration is ₹{int(predicted_price):,}")

    except Exception as e:
        st.error(f"Error: {e}")
