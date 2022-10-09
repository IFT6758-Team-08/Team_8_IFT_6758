import os, sys
import pandas as pd
import json



def get_game_events(data: dict):
    """
        This function takes the dictionary in the json files
        and returns one dictionary with two keys: game_id and play_by_play data.
    """
    events_dict = {} 
    for (key, info) in data.items():
        game_events_list = info['liveData']['plays']['allPlays']
        if len(game_events_list) != 0: #filters out the games where all keys are empty 
            events_dict["game_id"] = key
            events_dict["play_by_play"] = game_events_list
    return events_dict


def get_shots_goals_events(df):
    """
        This function takes a dataFrame of all events
        and keeps only Shots and Goals events
    """
    shots_goals_df = df.loc[(df['result.event'] == "Shot") | (df['result.event'] == "Goal")]
    return shots_goals_df


def filter_df(df):
    """
        This function filters a dataframe and only keeps a subset of columns
    """
    
    new_df = df[['game_id', 'about.periodTime', 'about.period',
                 'team.name', 'result.event', 'coordinates.x',
                 'coordinates.y', 'result.secondaryType',
                 'result.emptyNet', 'result.strength.name']]


    new_df_copy = new_df.copy()
    new_df_copy['shooter'] = df['players'].apply(return_shooter)
    new_df_copy['goalie'] = df['players'].apply(return_goalie)
    
    return new_df_copy
    

def return_goalie(list_players):
    """
        This function takes a list of dictionaries,
        each dictionary represents a player involved in a shot or a goal.
        It returns the name of the goalie, or none if there is no shooter
    """
    for player in list_players:
        if player['playerType'] == 'Goalie':
            return player['player']['fullName']
    return None
    
def return_shooter(list_players):
    """
        This function takes a list of dictionaries,
        each dictionary represents a player involved in a shot or a goal.
        It returns the name of the shooter, or none if there is no shooter
    """
    
    for player in list_players:
        if player['playerType'] == 'Shooter':  ##scorer in "goals" = shooter in shots??
            return player['player']['fullName']
    return None

def main():

    with open('2016_playoffs.json') as file:
        data = json.load(file)

    filtered_dict = get_game_events(data) #dictionary with 2 keys: game_id and play_by_play events

    df = pd.json_normalize(filtered_dict, record_path = ['play_by_play'],
                           meta = ['game_id']) #flattens the embedded dictionaries

    
    shots_goals_df = get_shots_goals_events(df) #only keeps the shots and goals events


    filtered_df = filter_df(shots_goals_df) # keeps only the columns we need
    filtered_df.to_csv('tidy_2.csv', index=False)

if __name__ == "__main__":
    main() 
