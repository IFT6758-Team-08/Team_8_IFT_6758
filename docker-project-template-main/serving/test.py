# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:45:21 2022

@author: LENOVO
"""

import pandas as pd
import requests
import json

def test_set_data(path):
    data = pd.read_csv(path)
    return data


def preprocess_data(df):
    # Convert rebound to 0 and 1 instead of True and False
    df['rebound'] = df['rebound'].astype(int)
    # Rearranging columns to make it easier to process data
    df=df[['game_id','period','period_time','coordinates_x','coordinates_y','shot_distance','shot_angle','secondary_type','last_event_type','time_from_last_event(s)','distance_from_last_event','rebound','angle_change','speed','last_event_coordinates_x','last_event_coordinates_y','goal']]
    # Convert period_time to seconds
    df['period_time'] = df['period_time'].apply(lambda x: int(x.split(':')[0])*60 + int(x.split(':')[1]))
    # Convert secondary_type and last_event_type to dummy variables
    df = pd.get_dummies(df, columns=['secondary_type','last_event_type'], drop_first=True)
    return df

df_regular_f2 = test_set_data("M2_added_features_all_test_regular.csv")
df_regular_f2 = preprocess_data(df_regular_f2)
X_regular_f2 = df_regular_f2[['shot_angle','speed','shot_distance','distance_from_last_event','period_time','coordinates_y','time_from_last_event(s)']]


r = requests.post(
 	"http://localhost:5010/predict", 
 	json=X_regular_f2.to_json(orient = 'table')
)

print(r.json())
