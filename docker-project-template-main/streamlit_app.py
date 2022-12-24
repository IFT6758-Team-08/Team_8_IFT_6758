import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ift6758.ift6758.client.game_client as gc
from ift6758.ift6758.client.serving_client import ServingClient
import json


"""
General template for your streamlit app. 
Feel free to experiment with layout and adding functionality!
Just make sure that the required functionality is included as well
"""

sc = ServingClient()

st.title("Hockey Visualization App")

model = 'xgb3'

with st.sidebar:
    # Dropdown for Workspace selection
    workspace = st.selectbox("Workspace", ["rachel98"])
    # Dropdown for Model selection
    m = st.selectbox("Model", ["xgb3"])
    # Dropdown for Version selection
    version = st.selectbox("Version", ["1.0.2", "1.0.1", "1.0.0"])
    get_model = st.button("Get Model")
    # print(model)
    if get_model:
        #todo: call download model func

        model = m
        a = sc.download_registry_model(workspace, model, version)
        # print(a)
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
        # st.write("Game data")
        res, away, home, cur_period, remaining_time, score = ping
        st.write("Away Team: ", away)
        st.write("Home Team: ", home)
        st.write("Current Period: ", str(cur_period))
        st.write("Remaining Time: ", remaining_time)
        st.text("Scores: ")
        st.write("away: ", str(score['away']))
        st.write("home: ", str(score['home']))
        # st.write(res)

        print("model is", model)
        if type(res) is not str:
            if model == 'xgb3':
                features = ['shot_angle', 'speed', 'shot_distance', 'distance_from_last_event', 'period_time',
                            'coordinates_y', 'time_from_last_event(s)']
                teams = res['team']
                df_to_predict = res[features]
            # elif model == 'xgb2':
            #     features = ['period', 'period_time', 'coordinates_x', 'coordinates_y', 'shot_distance', 'shot_angle',
            #                 'secondary_type', 'last_event_type', 'time_from_last_event(s)', 'distance_from_last_event',
            #                 'rebound', 'angle_change', 'speed', 'last_event_coordinates_x', 'last_event_coordinates_y',
            #                 'goal']
            #     df_to_predict = res[features]
            # elif model == 'xgb1':
            #     df_to_predict = res[:]

            # print(df_to_predict)

            predicted = sc.predict(df_to_predict)
            predicted = json.loads(predicted)
            # st.write(predicted)

            # st.write("using model ", predicted['model']['0'], " to predict...")
            # st.write(predicted['pred_proba'].values())


            df_to_predict['probability'] = predicted['pred_proba'].values()

            df_with_teams = df_to_predict.copy()
            df_with_teams['team'] = list(teams)
            away_prob = sum(df_with_teams[df_with_teams['team'] == away]['probability'])
            home_prob = sum(df_with_teams[df_with_teams['team'] == home]['probability'])

            st.write("home team xG so far: ", home_prob, 'diff: ', home_prob - score['home'])
            st.write("away team xG so far: ", away_prob, 'diff: ', away_prob - score['away'])
            st.text("data used for prediction and probability")
            st.write(df_to_predict)

        else:
            st.write(res)




