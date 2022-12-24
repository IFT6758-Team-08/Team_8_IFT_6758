import json
import requests
import pandas as pd
import numpy as np
import math
import copy
import os


def get_data(GAME_ID):
    url = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(GAME_ID)
    r = requests.get(url).json()
    game = {}
    if r.get('liveData') is not None:  # if 'liveData' == None, then no data for that game.
        game['gameData'], game['liveData'] = r['gameData'],r['liveData']
        game['liveData']['plays']['allPlays'] = game['liveData']['plays']['allPlays'][:]
        return game
    else:
        return "Game Id data not available!"




def get_game_events(data: dict):
    """
        This function takes a dictionary with a game's information (gameData and liveData)
        and returns two dictionaries:
            games_metadata: key=game_ID, values = dictionary with the rinkSide of each team for every period
            events_dict: key=game_ID, value =  all play by play events
    """
    events_dict, games_metadata = {}, {}

    # Iterating through all the games of a json file

    game_meta, events_list = {}, []

    # getting a dictionary with the rinkSide of each team for a game
    period_info_list = data['liveData']['linescore']['periods']  # list containing the team's side in each period
    if len(period_info_list) > 0:
        game_meta[data['gameData']['teams']['home']['name']], game_meta[
            data['gameData']['teams']['away']['name']] = get_metadata_rinkSide(period_info_list)

    # getting a dictionary with all the "shot" or "goal" event for a specific game:
    for event in data['liveData']['plays']['allPlays']:
        events_list.append(event)
        # if event['result']['event'] == "Shot" or event['result'][
        #     'event'] == "Goal":  # filters all other events and games with no data
        #     events_list.append(event)

    # Adding the game's data to the whole season dictionaries
    # games_metadata[key] = game_meta
    # events_dict[key] = events_list
    return events_list, game_meta


def get_metadata_rinkSide(period_info_list):
    """
    This function takes a list with the period's information of a game
    and returns the rinkSide of both teams (home and away) in each period.
    """

    game_period_home, game_period_away = {}, {}

    # Iterating through all periods of a game
    for i, period in enumerate(period_info_list):
        if 'rinkSide' in period['home']:
            game_period_home[str(i + 1)] = period['home']['rinkSide']
            game_period_away[str(i + 1)] = period['away']['rinkSide']


    return game_period_home, game_period_away


def filter_df(df, games_metadata):
    """
        This function filters a dataframe and only keeps a subset of columns
    """
    flag = True
    new_df = pd.DataFrame()
    # Keeping only the necessary columns:
    # print('Hit' in list(df['result.event']))
    if ('Shot' in list(df['result.event'])) or ("Goal" in list(df['result.event'])):
        new_df[['period', 'period_time',
                'team', 'event', 'coordinates_x',
                'coordinates_y']] = df[['about.period', 'about.periodTime',
                                        'team.name', 'result.event', 'coordinates.x',
                                        'coordinates.y']]
        new_df['empty_net'] = np.nan
        new_df['secondary_type'] = np.nan

        if 'result.emptyNet' in df.keys():
            new_df['empty_net'] = df['result.emptyNet']
        if 'result.secondaryType' in df.keys():
            new_df['secondary_type'] = df['result.secondaryType']
    else:
        new_df[['period', 'period_time',
                'team', 'event', 'coordinates_x',
                'coordinates_y']] = df[['about.period', 'about.periodTime',
                                    'team.name', 'result.event', 'coordinates.x',
                                    'coordinates.y']]
        flag = False
    # return "no shot", 0, 0
    # print(new_df)
    # print(new_df["event"])
    last_event_idx = new_df.index[-1]

    new_df = new_df[new_df.coordinates_x.notna() & new_df.coordinates_y.notna()]
    last_valid_event_idx = new_df.index[-1]
    new_df = new_df.reset_index(drop=True)
    new_df_copy = new_df.copy()
    # print(new_df_copy)
    new_df_copy['rinkSide'] = new_df.apply(lambda x: get_rinkside(x['team'], x['period'], games_metadata),
                                           axis=1)
    # print(new_df["secondary_type"])
    # print(new_df_copy)
    return new_df_copy, last_event_idx, last_valid_event_idx, flag


def get_rinkside(team, period, games_metadata):
    """
        This function return the rinkside of "team" in a specific period of a game.
    """
    try:
        # All odd periods have the same rinkSide
        if period % 2 == 1:
            return games_metadata[team]["1"]
        elif period % 2 == 0:
            return games_metadata[team]["2"]
    except:
        return None


def tidy_data(data):

    """
    returns the desired dataframe for the given year(season)
    """
    # getting two dictionaries: play_by_play events (filtered_dict) and a rinkSide information for teams (game_metadata)
    filtered_dict, games_metadata = get_game_events(data)

    all_df = pd.DataFrame()

    df = pd.json_normalize(filtered_dict)  # flattens the embedded dictionaries
    # print(df['liveData']['plays']['allPlays'])
    all_df = pd.concat([all_df, df])
    all_df.reset_index(inplace=True, drop=True)

    filtered_df, last_event_idx, last_valid_event_idx, flag = filter_df(all_df, games_metadata)  # keeps only the columns we need

    return filtered_df, last_event_idx, last_valid_event_idx, flag


def preprocess(df_all_season):
    # print(df_all_season)
    # print(df_all_season['rinkSide'])
    df_all_season = df_all_season[df_all_season.rinkSide.notna() & df_all_season.coordinates_x.notna() & df_all_season.coordinates_y.notna()].reset_index(drop=True)
    # print(df_all_season.empty)
    # print("im here now")
    if df_all_season.empty:
        return 'Dataframe is missing some main features!'
    df_all_season["offensive_goal_post_x"] = [-89 if i=="right" else 89 for i in df_all_season["rinkSide"]]
    df_all_season["offensive_goal_post_y"] = 0
    # print(df_all_season)
    df_all_season["shot_distance"] = df_all_season.apply(lambda x: np.sqrt((x['offensive_goal_post_x']-x['coordinates_x'])**2 + (x['offensive_goal_post_y']-x['coordinates_y'])**2), axis=1)
    df_all_season['goal'] = np.where(df_all_season['event']=="Goal", 1, 0)
    df_all_season['shot_angle'] = df_all_season.apply(lambda x: np.arcsin((x['offensive_goal_post_y']-x['coordinates_y'])/(x['shot_distance']+0.0001))*180/math.pi, axis=1)
    df_all_season["empty_net"] = df_all_season["empty_net"].replace(np.nan, False)
    df_all_season['is_empty_net'] = np.where(df_all_season['empty_net']==True, 1, 0)
    return df_all_season


def preprocess_data2(df):
    # Convert rebound to 0 and 1 instead of True and False
    df['rebound'] = df['rebound'].astype(int)
    # Rearranging columns to make it easier to process data
    df = df[['team', 'period', 'period_time', 'coordinates_x', 'coordinates_y', 'shot_distance', 'shot_angle',
             'secondary_type', 'last_event_type', 'time_from_last_event(s)', 'distance_from_last_event', 'rebound',
             'angle_change', 'speed', 'last_event_coordinates_x', 'last_event_coordinates_y', 'goal']]
    # Convert period_time to seconds
    df['period_time'] = df['period_time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    # Convert secondary_type and last_event_type to numeric
    # df['secondary_type'] = df['secondary_type'].astype('category').cat.codes
    # df['last_event_type'] = df['last_event_type'].astype('category').cat.codes
    # Convert secondary_type and last_event_type to dummy variables
    df = pd.get_dummies(df, columns=['secondary_type', 'last_event_type'], drop_first=True)
    all_dummies = ['secondary_type_Deflected', 'secondary_type_Slap Shot',
    'secondary_type_Snap Shot', 'secondary_type_Tip-In',
    'secondary_type_Wrap-around', 'secondary_type_Wrist Shot',
    'last_event_type_Faceoff', 'last_event_type_Giveaway',
    'last_event_type_Goal', 'last_event_type_Hit',
    'last_event_type_Missed Shot', 'last_event_type_Penalty',
    'last_event_type_Shot', 'last_event_type_Takeaway']
    for f in all_dummies:
        if f not in df.keys():
            df[f] = 0

    # df = df[['team', 'period', 'period_time', 'coordinates_x', 'coordinates_y',
    #    'shot_distance', 'shot_angle', 'time_from_last_event(s)',
    #    'distance_from_last_event', 'rebound', 'angle_change', 'speed',
    #    'last_event_coordinates_x', 'last_event_coordinates_y', 'goal',
    #    'secondary_type_Deflected', 'secondary_type_Slap Shot',
    #    'secondary_type_Snap Shot', 'secondary_type_Tip-In',
    #    'secondary_type_Wrap-around', 'secondary_type_Wrist Shot',
    #    'last_event_type_Faceoff', 'last_event_type_Giveaway',
    #    'last_event_type_Goal', 'last_event_type_Hit',
    #    'last_event_type_Missed Shot', 'last_event_type_Penalty',
    #    'last_event_type_Shot', 'last_event_type_Takeaway']]
    return df
################feature eng 2###########################
def filter_features(prev_events_df, goal_shot_df):
    # print(prev_events_df)
    # print(goal_shot_df)
    goal_shot_df = goal_shot_df[["team", "period", "period_time", "coordinates_x", "coordinates_y", "shot_distance", "shot_angle", "secondary_type", "goal"]]
    new_df, last_shot_idx = add_prev_event_info(goal_shot_df, prev_events_df)
    new_df = new_df[new_df.last_event_type.notna()].reset_index(drop=True)

    new_df = create_more_features(new_df)

    return new_df, last_shot_idx

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

def add_prev_event_info(goal_shot_df, prev_events_df):
    # prev_events_df = pd.read_csv(path)
    # print(goal_shot_df)
    # shot_idx = goal_shot_df.index
    # print(prev_events_df)
    shot_idx = goal_shot_df.index
    last_shot_idx =shot_idx[-1]
    goal_shot_df.reset_index(inplace=True, drop=True)
    # print(goal_shot_df)
    new_df = goal_shot_df.copy()
    # print(new_df.index)
    # print(prev_events_df.index)

    # print(shot_idx)
    for i in range(len(goal_shot_df)):
        shot = goal_shot_df.iloc[i]
        period, period_time, x, y = shot["period"], shot["period_time"], shot["coordinates_x"], shot["coordinates_y"]

        current_event_index = shot_idx[i]

        # print(current_event_index)
        if current_event_index > 0:
            prev_event_index = current_event_index - 1
            # print(prev_event_index)
            prev_event = prev_events_df.iloc[prev_event_index]
            # print(prev_event['event'])
            new_df.loc[i, ["last_event_type", "last_event_coordinates_x", "last_event_coordinates_y"]] = list(prev_event[['event', 'coordinates_x', 'coordinates_y']])
            # print(new_df)
            new_df.loc[i, ["time_from_last_event(s)"]] = time_diff(period_time,  prev_event["period_time"])
            new_df.loc[i, ["distance_from_last_event"]] = events_distance((x, y), (prev_event["coordinates_x"], prev_event["coordinates_y"]))
            # print(new_df)
        else:
            print("it was the first event. No previous event is available")

    return new_df, last_shot_idx

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


def get_new_events_only(all_data, tracker_game):
    # print(data.keys())

    live_data = all_data['liveData']['plays']['allPlays']
    last_val_idx = tracker_game["last_val_idx"] #the previous event we can use for our first new shot
    last_idx = tracker_game["last_idx"] #last event we checked
    last_val_event = live_data[last_val_idx]
    new_live_data = live_data[last_val_idx:]
    all_data['liveData']['plays']['allPlays'] = new_live_data
    return all_data, last_val_event


def ping_game_client(GAME_ID):

    data = get_data(GAME_ID)
    if data == "Game Id data not available!":
        return "Game Id data not available!", "N/A", "N/A", "N/A", "N/A", "N/A"
    else:
        away_team = data['liveData']['boxscore']['teams']['away']['team']['name']
        home_team = data['liveData']['boxscore']['teams']['home']['team']['name']
        cur_period = data['liveData']['plays']['allPlays'][-1]['about']['period']
        remaining_time = data['liveData']['plays']['allPlays'][-1]['about']['periodTimeRemaining']
        score = data['liveData']['plays']['allPlays'][-1]['about']['goals']

        if os.path.exists("tracker.json"):
            with open("tracker.json", "r") as file:
                old_tracker = json.load(file)
        else:
            old_tracker = {}

        if GAME_ID in old_tracker:
            old_tracker_game = old_tracker[GAME_ID]
            prev_last_valid_event_idx = old_tracker_game["last_val_idx"]
            prev_last_event_idx = old_tracker_game["last_idx"]
            prev_away_prob = old_tracker_game['away_prob']
            prev_home_prob = old_tracker_game['home_prob']

            number_of_all_events = len(data['liveData']['plays']['allPlays'])
            if prev_last_event_idx + 1 == number_of_all_events:
                # print("We have already checked all the available events!")
                return "We have already checked all the available events!", away_team, home_team, cur_period, remaining_time, score
            else:
                data1 = copy.deepcopy(data)
                # print(len(data['liveData']['plays']['allPlays']))
                data, last_val_event = get_new_events_only(data1, old_tracker_game)
                # print(data['liveData']['plays']['allPlays'][0])
                # print("@@@@@@@@@@@@")
        else:
            prev_last_valid_event_idx = 0
            prev_last_event_idx = 0
            prev_away_prob = 0
            prev_home_prob = 0

        # print(data['liveData']['plays']['allPlays'][0])
        df, last_event_idx, last_valid_event_idx, flag = tidy_data(data)

        if flag: #if there was at least one shot
            all_events = preprocess(df)
            if type(all_events) is str:
                return all_events, away_team, home_team, cur_period, remaining_time, score
            else:
                goal_shot_df = all_events[(all_events['event'] == 'Goal') | (all_events['event'] == 'Shot')]
                my_df, last_shot_idx = filter_features(all_events, goal_shot_df)
                final_df = preprocess_data2(my_df)

            # print(final_df)
            # print(final_df.keys())
        else:
            # print("No new shot was found!")
            return "No new shot was found!", away_team, home_team, cur_period, remaining_time, score
        # print(last_valid_event_idx, prev_last_valid_event_idx)
        tracker = {"last_val_idx": int(last_valid_event_idx + prev_last_valid_event_idx), "last_idx": int(last_event_idx + prev_last_valid_event_idx), 'away_prob': prev_away_prob, 'home_prob': prev_home_prob}


        old_tracker[GAME_ID] = tracker

        with open("tracker.json", "w") as file:
            json.dump(old_tracker, file)
        # print(final_df.iloc[0])
        return final_df, away_team, home_team, cur_period, remaining_time, score

# GAME_ID = "2021020324"
# a=ping_game_client(GAME_ID)
# print(a)
