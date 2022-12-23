import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ift6758.ift6758.client.game_client as gc
from ift6758.ift6758.client.serving_client import ServingClient
"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

sc = ServingClient()

st.title("Hockey Visualization App")

with st.sidebar:
    # Dropdown for Workspace selection
    workspace = st.selectbox("Workspace", ["rachel98"])
    # Dropdown for Model selection
    model = st.selectbox("Model", ["xgb1", "XGBoost 2", "XGBoost 3"])
    # Dropdown for Version selection
    version = st.selectbox("Version", ["1.0.0", "v2", "v3"])
    get_model = st.button("Get Model")
    if get_model:
        #todo: call download model func
        a = sc.download_registry_model(workspace, model, version)
        print(a)
        st.write("model downloaded")

with st.container():
    # Game ID as dropdown
    # game_id = st.selectbox("Game ID", ["2021020329", "2", "3"])
    game_id = st.text_input('Game ID', "2021020329")
    st.write('Game ID selected: ', game_id)
    # Add button to load data
    load_data = st.button("Ping Game")

    # Display game data if button is pressed
    if load_data:
        ping = gc.ping_game_client(game_id)
        st.write("Game data")
        st.write(ping)




