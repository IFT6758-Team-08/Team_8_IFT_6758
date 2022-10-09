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
        # print(game_events_list)
        if len(game_events_list) != 0:  # filters out the games where all keys are empty
            # events_dict["game_id"] = key
            events_dict[key] = game_events_list
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

def tidy_all_seasons(path, year, to_year):
    """
    returns the dataframe for all the seasons(from year to to_year)
    """
    all_df = pd.DataFrame()
    for y in range(year, to_year + 1):
        file_path_regular, file_path_playoff = path + "/" + str(y) + "_regular_season.json", path + "/" + str(y) + "_playoffs.json"
        df_regular = tidy_one_season(file_path_regular)
        df_playoff = tidy_one_season(file_path_playoff)
        all_df = pd.concat([all_df, df_regular, df_playoff])
    return all_df


def tidy_one_season(path):
    """
    returns the desired dataframe for the given year(season)
    """

    with open(path) as file:
        data = json.load(file)
    filtered_dict = get_game_events(data)  # dictionary with 2 keys: game_id and play_by_play events
    game_ids = filtered_dict.keys()
    all_df = pd.DataFrame()
    for game_id in game_ids:
        df = pd.json_normalize(filtered_dict[game_id])  # flattens the embedded dictionaries
        df['game_id'] = game_id
        all_df = pd.concat([all_df, df])
    all_df.reset_index(inplace=True, drop=True)

    shots_goals_df = get_shots_goals_events(all_df)  # only keeps the shots and goals events
    filtered_df = filter_df(shots_goals_df)  # keeps only the columns we need
    return filtered_df


def main():
    all_df = tidy_all_seasons("get data", 2016, 2021)
    all_df.to_csv('tidy_df.csv', index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()