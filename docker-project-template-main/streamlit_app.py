import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ift6758.ift6758.client.game_client as gc
from ift6758.ift6758.client.serving_client import ServingClient
import json
from comet_ml import API
import os

# """
# General template for your streamlit app. 
# Feel free to experiment with layout and adding functionality!
# Just make sure that the required functionality is included as well
# """

sc = ServingClient()

st.title("Hockey Visualization App")

model = 'xgb2'
# model_ver = {'xgb3':["1.0.2", "1.0.1", "1.0.0"], 'xgb2':['1.0.0'], 'xgb1':['1.0.0']}
api = API(api_key = os.environ.get('COMET_API_KEY'))

with st.sidebar:
    # Dropdown for Workspace selection
    workspace = st.selectbox("Workspace", ["rachel98"])
    # Dropdown for Model selection
    m = st.selectbox("Model", ["xgb2", 'xgb3'])
    ver = api.get_registry_model_versions(workspace, m)
    # Dropdown for Version selection
    version = st.selectbox("Version", ver)
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
        st.write(game_id)
        ping = gc.ping_game_client(game_id)
        # st.write("Game data")
        res, away, home, cur_period, remaining_time, score = ping
        # st.write("Away Team: ", away)
        # st.write("Home Team: ", home)
        st.subheader(home+"(home) VS "+away+"(away)")
        st.write("Period ", str(cur_period), " - ", remaining_time, " left")
        # st.write("Remaining Time: ", remaining_time)
        # st.text("Scores: ")
        # st.write("away: ", str(score['away']))
        # st.write("home: ", str(score['home']))
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
            elif model == 'xgb2':
                features = ['period', 'period_time', 'coordinates_x', 'coordinates_y',
       'shot_distance', 'shot_angle', 'time_from_last_event(s)',
       'distance_from_last_event', 'rebound', 'angle_change', 'speed',
       'last_event_coordinates_x', 'last_event_coordinates_y', 'goal',
       'secondary_type_Deflected', 'secondary_type_Slap Shot',
       'secondary_type_Snap Shot', 'secondary_type_Tip-In',
       'secondary_type_Wrap-around', 'secondary_type_Wrist Shot',
       'last_event_type_Faceoff', 'last_event_type_Giveaway',
       'last_event_type_Goal', 'last_event_type_Hit',
       'last_event_type_Missed Shot', 'last_event_type_Penalty',
       'last_event_type_Shot', 'last_event_type_Takeaway']

                teams = res['team']
                df_to_predict = res[features]

                # st.write(df_to_predict)

            predicted = sc.predict(df_to_predict)
            # st.write(predicted)
            predicted = json.loads(predicted)
            # st.write(predicted)

            # st.write("using model ", predicted['model']['0'], " to predict...")
            # st.write(predicted['pred_proba'].values())


            df_to_predict['probability'] = predicted['pred_proba'].values()

            df_with_teams = df_to_predict.copy()
            df_with_teams['team'] = list(teams)
            away_prob = sum(df_with_teams[df_with_teams['team'] == away]['probability'])
            home_prob = sum(df_with_teams[df_with_teams['team'] == home]['probability'])

            # st.write(home_prob, 'diff: ', home_prob - score['home'])
            with open("tracker.json", 'r') as file:
                prev_prob = json.load(file)
                prev_home_prob = prev_prob[game_id]['home_prob']
                prev_away_prob = prev_prob[game_id]['away_prob']


            col1, col2 = st.columns(2)
            # st.write(round(home_prob, 2))
            col1.metric(label=home+" xG (actual)", value=str(round(home_prob + prev_home_prob, 2)) + "(" +str(score['home']) + ")", delta=round(home_prob - score['home'], 2))
            col2.metric(label=away+" xG (actual)", value=str(round(away_prob + prev_away_prob, 2)) + "(" +str(score['away']) + ")", delta=round(away_prob - score['away'], 2))

            prev_prob[game_id]['home_prob'] = round(home_prob + prev_home_prob, 2)
            prev_prob[game_id]['away_prob'] = round(away_prob + prev_away_prob, 2)
            with open("tracker.json", 'w') as file:
                json.dump(prev_prob, file)

            st.text("data used for prediction and probability")
            st.write(df_to_predict)

        else:
            st.write(res)




