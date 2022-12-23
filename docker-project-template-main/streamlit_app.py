import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

st.title("Hockey Visualization App")

with st.sidebar:
    # Dropdown for Workspace selection
    workspace = st.selectbox("Workspace", ["rachel98"])
    # Dropdown for Model selection
    model = st.selectbox("Model", ["XGBoost 1", "XGBoost 2", "XGBoost 3"])
    # Dropdown for Version selection
    version = st.selectbox("Version", ["v1", "v2", "v3"])

with st.container():
    # Game ID as dropdown
    game_id = st.selectbox("Game ID", ["1", "2", "3"])
    # Add button to load data
    load_data = st.button("Ping Game")
    # Display game data if button is pressed
    if load_data:
        st.write("Game data")




