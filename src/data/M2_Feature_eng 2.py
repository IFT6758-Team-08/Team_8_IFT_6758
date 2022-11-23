import numpy as np
import pandas as pd
from comet_ml import Experiment
import os

def split_dataframe(path):
    df = pd.read_csv(path)
    # for year in range(from_year, to_year+1):
    #     year_df = df[df["season"] == year]
    # year_df.to_csv("M2_tidy_data_"+str(year)+".csv", index=False, encoding='utf-8-sig')
    filter_features(df)

def filter_features(df):
    new_df = df[["game_id", "period", "period_time", "coordinates_x", "coordinates_y", "shot_distance", "shot_angle", "secondary_type", "goal"]]
    new_df = add_prev_event_info(new_df, "M2_regular_tidy_data.csv")
    # For test data 
    # new_df = add_prev_event_info(new_df, "M2_regular_2020_cleaned1.csv")
    # new_df = add_prev_event_info(new_df, "M2_playoff_2020_cleaned1.csv")
    new_df = create_more_features(new_df)
    #new_df.to_csv("M2_added_features_all_test_regular.csv")
    # new_df.to_csv("M2_added_features_all_test_playoff.csv")
    new_df.to_csv("M2_added_features_all.csv")
    # print(new_df.head(15))

def time_diff(time1, time2):
    t1 = time1.split(":")
    t2 = time2.split(":")
    t1_seconds = int(t1[0]) * 60 + int(t1[1])
    t2_seconds = int(t2[0]) * 60 + int(t2[1])
    if (t1_seconds - t2_seconds) < 0:
        # print(time1, time2)
        print("time difference is less than 0")
    return t1_seconds - t2_seconds


def events_distance(point1, point2):
    dist = np.linalg.norm(np.array(point1) - np.array(point2))
    return dist

def add_prev_event_info(df, path):
    prev_events_df = pd.read_csv(path)
    new_df = df.copy()
    for i in range(len(df)):
        shot = df.iloc[i]
        gameid, period, period_time, x, y = shot["game_id"], shot["period"], shot["period_time"], shot["coordinates_x"], shot["coordinates_y"]
        current_event = prev_events_df[prev_events_df["game_id"] == gameid]
        current_event = current_event[current_event["period"] == period]
        current_event = current_event[current_event["period_time"] == period_time]
        current_event = current_event[current_event["coordinates_x"] == x]
        current_event = current_event[current_event["coordinates_y"] == y]
        current_event_index = current_event.index[0]
        if current_event_index > 0:
            prev_event_index = current_event_index - 1
            prev_event = prev_events_df.iloc[prev_event_index]
            new_df.loc[i, ["last_event_type", "last_event_coordinates_x", "last_event_coordinates_y"]] = list(prev_event[['event', 'coordinates_x', 'coordinates_y']])
            # print(new_df.iloc[i])
            new_df.loc[i, ["time_from_last_event(s)"]] = time_diff(period_time,  prev_event["period_time"])
            new_df.loc[i, ["distance_from_last_event"]] = events_distance((x, y), (prev_event["coordinates_x"], prev_event["coordinates_y"]))
            # print(new_df)
        else:
            print("it was the first event. No previous event is available")
    return new_df

def create_more_features(df):
    new_df = df.copy()
    for i in range(len(df)):
        row = df.iloc[i]
        rebound = add_rebound(row)
        new_df.loc[i, ["rebound"]] = rebound
        if rebound:
            angle_change = add_angle_change(row, df.iloc[i-1])
        else:
            angle_change = 0
        new_df.loc[i, ["angle_change"]] = angle_change
        speed = add_speed(row)
        new_df.loc[i, ["speed"]] = speed
    return new_df

def add_rebound(row):
    if row["last_event_type"] == "Shot":
        return True
    else:
        return False


def add_angle_change(row, prev_row):
    prev_shot_angle = prev_row["shot_angle"]
    current_shot_angle = row["shot_angle"]
    sign = row["coordinates_x"] * row["last_event_coordinates_x"]
    if sign >= 0: #same rink side
        angle_change = abs(prev_shot_angle - current_shot_angle)

    else: #not in the same rink side
        angle_change = 180 - abs(prev_shot_angle + current_shot_angle)

    return angle_change

def add_speed(row):
    dist = row["distance_from_last_event"]
    time = row["time_from_last_event(s)"]
    if time == 0:
        time += 0.0000001
    return dist/time


def get_subset_df(path):  #path to the final feature eng 2 datafram csv file
    df = pd.read_csv(path, index_col=0)
    #get data for gameid = 2017021065
    subset_df = df[df["game_id"] == 2017021065]
    return subset_df.reset_index(drop=True)


def comet(path):
    subset_df = get_subset_df(path)
    experiment = Experiment(
        api_key=os.environ.get('COMET_API_KEY'),
        project_name='ift-6758-team-8',
        workspace="Rachel98",
    )
    experiment.log_dataframe_profile(
        subset_df,
        name='wpg_v_wsh_2017021065',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
    )


# import time
# start_time = time.time()
# For test data
# split_dataframe("D:/NHLPro/data/test_data_playoff.csv") # Path of the Feature Engineering 1 output file
# print("--- %s seconds ---" % (time.time() - start_time))

comet("M2_added_features_all.csv")
