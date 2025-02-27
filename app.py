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

# --- Handle CPU Brand Issue ---
if 'Cpu brand' in df.columns:
    df = df.dropna(subset=['Cpu brand'])  # Remove NaN values
    df['Cpu brand'] = df['Cpu brand'].astype(str).str.strip()
    cpu_options = sorted(df['Cpu brand'].unique().tolist())

    # If "Apple M1" is missing, default to "Intel Core i5"
    mac_cpu = "Apple M1" if "Apple M1" in cpu_options else "Intel Core i5"
else:
    st.error("Error: 'Cpu brand' column not found in DataFrame.")
    cpu_options = ["Unknown"]
    mac_cpu = "Intel Core i5"

# --- Handle GPU Brand Issue ---
if 'Gpu brand' in df.columns:
    df = df.dropna(subset=['Gpu brand'])
    df['Gpu brand'] = df['Gpu brand'].astype(str).str.strip()
    gpu_options = sorted(df['Gpu brand'].unique().tolist())
else:
    st.error("Error: 'Gpu brand' column not found in DataFrame.")
    gpu_options = ["Unknown"]

# --- Predefined Configurations ---
configurations = {
    "Windows i3": {'cpu': 'Intel Core i3', 'gpu': 'Intel', 'ram': 8, 'ssd': 256, 'hdd': 0, 'os': 'Windows'},
    "Windows i5": {'cpu': 'Intel Core i5', 'gpu': 'Intel', 'ram': 16, 'ssd': 512, 'hdd': 0, 'os': 'Windows'},
    "Windows i7": {'cpu': 'Intel Core i7', 'gpu': 'Nvidia', 'ram': 32, 'ssd': 1000, 'hdd': 0, 'os': 'Windows'},
    "MacBook Air": {'cpu': mac_cpu, 'gpu': 'Apple', 'ram': 8, 'ssd': 512, 'hdd': 0, 'os': 'MacOS'}
}

# --- Button Layout ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Windows i3"):
        st.session_state.update(configurations["Windows i3"])
with col2:
    if st.button("Windows i5"):
        st.session_state.update(configurations["Windows i5"])
with col3:
    if st.button("Windows i7"):
        st.session_state.update(configurations["Windows i7"])
with col4:
    if st.button("MacBook Air"):
        st.session_state.update(configurations["MacBook Air"])

# --- User Inputs ---
cpu = st.selectbox('CPU', cpu_options, key='cpu')
gpu = st.selectbox('GPU', gpu_options, key='gpu')
ram = st.slider('RAM (GB)', 4, 64, st.session_state.get('ram', 8), step=4)
ssd = st.number_input('SSD (GB)', min_value=0, max_value=2000, value=st.session_state.get('ssd', 256), step=128)
hdd = st.number_input('HDD (GB)', min_value=0, max_value=2000, value=st.session_state.get('hdd', 0), step=128)
os = st.selectbox('Operating System', ['Windows', 'MacOS', 'Linux'], key='os')

# --- Prediction ---
if st.button("Predict Price"):
    try:
        # Prepare input array
        query = np.array([[cpu, gpu, ram, ssd, hdd, os]])
        predicted_price = pipe.predict(query)[0]
        st.success(f"Predicted Price: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

st.write("🔹 **Available CPU Brands:**", cpu_options)
st.write("🔹 **Available GPU Brands:**", gpu_options)
