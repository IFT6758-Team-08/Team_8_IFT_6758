import os, sys
import pandas as pd
import json


def get_game_events(data: dict):
    """
        This function takes a dictionary with a game's information (gameData and liveData)
        and returns two dictionaries:
            games_metadata: key=game_ID, values = dictionary with the rinkSide of each team for every period
            events_dict: key=game_ID, value =  all play by play events
    """
    events_dict, games_metadata = {}, {}

    # Iterating through all the games of a json file
    for (key, info) in data.items():
        game_meta, events_list = {}, []

        # getting a dictionary with the rinkSide of each team for a game
        period_info_list = info['liveData']['linescore']['periods']  # list containing the team's side in each period
        if len(period_info_list) > 0:
            game_meta[info['gameData']['teams']['home']['name']], game_meta[
                info['gameData']['teams']['away']['name']] = get_metadata_rinkSide(period_info_list)

        # getting a dictionary with all the "shot" or "goal" event for a specific game:
        for event in info['liveData']['plays']['allPlays']:
            events_list.append(event)
            # if event['result']['event'] == "Shot" or event['result'][
            #     'event'] == "Goal":  # filters all other events and games with no data
            #     events_list.append(event)

        # Adding the game's data to the whole season dictionaries
        games_metadata[key] = game_meta
        events_dict[key] = events_list
    return events_dict, games_metadata


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


def filter_df(df):
    """
        This function filters a dataframe and only keeps a subset of columns
    """
    new_df = pd.DataFrame()
    # print(df)
    # Keeping only the necessary columns:
    new_df[['game_id', 'period', 'period_time',
            'team', 'event', 'coordinates_x',
            'coordinates_y', 'secondary_type', 'penalty_minutes']] = df[['game_id', 'about.period', 'about.periodTime',
                                                 'team.name', 'result.event', 'coordinates.x',
                                                 'coordinates.y', 'result.secondaryType', 'result.penaltyMinutes']]

    # new_df = new_df[new_df.coordinates_x.notna() & new_df.coordinates_y.notna()].reset_index(drop=True)

    # Adding rinkSide, shooter's and goalie's name to each event
    new_df_copy = new_df.copy()
    # new_df_copy['rinkSide'] = new_df.apply(lambda x: get_rinkside(x['game_id'], x['team'], x['period'], games_metadata),
    #                                        axis=1)
    # new_df_copy['shooter'] = df['players'].apply(get_shooter)
    # new_df_copy['goalie'] = df['players'].apply(get_goalie)

    return new_df_copy


def get_rinkside(game_id, team, period, games_metadata):
    """
        This function return the rinkside of "team" in a specific period of a game.
    """
    try:
        # All odd periods have the same rinkSide
        if period % 2 == 1:
            return games_metadata[game_id][team]["1"]
        elif period % 2 == 0:
            return games_metadata[game_id][team]["2"]
    except:
        return None


def get_goalie(list_players):
    """
        This function takes a list of dictionaries,
        each dictionary represents a player involved in a shot or a goal.
        It returns the name of the goalie, or none if there is no shooter
    """
    for player in list_players:
        if player['playerType'] == 'Goalie':
            return player['player']['fullName']
    return None


def get_shooter(list_players):
    """
        This function takes a list of dictionaries,
        each dictionary represents a player involved in a shot or a goal.
        It returns the name of the shooter, or none if there is no shooter
    """

    for player in list_players:
        if player['playerType'] == 'Shooter' or player['playerType'] == 'Scorer':
            return player['player']['fullName']
    return None


def tidy_one_season(path):
    """
    returns the desired dataframe for the given year(season)
    """
    # open the json file generated in data acquisition
    with open(path) as file:
        data = json.load(file)

    # getting two dictionaries: play_by_play events (filtered_dict) and a rinkSide information for teams (game_metadata)
    filtered_dict, games_metadata = get_game_events(data)
    game_ids = filtered_dict.keys()
    all_df = pd.DataFrame()

    # converting the dictionaries to a dataframe:
    for game_id in game_ids:
        df = pd.json_normalize(filtered_dict[game_id])  # flattens the embedded dictionaries
        df['game_id'] = game_id
        all_df = pd.concat([all_df, df])
    all_df.reset_index(inplace=True, drop=True)

    # shots_goals_df = get_shots_goals_events(all_df)  # only keeps the shots and goals events
    filtered_df = filter_df(all_df)  # keeps only the columns we need
    return filtered_df


def tidy_all_seasons(path, year, to_year):
    """
    returns one dataframe per season and saves them in a csv file.
    """
    for y in range(year, to_year + 1):
        file_path_regular= path + "/" + str(y) + "_regular_season.json"
        output_file = "M2_regular_bonus_" + str(y) + "_cleaned.csv"

        # tidying the data for regular and playoff season in two dataframes
        print("tidying regular season ", y)
        df_regular = tidy_one_season(file_path_regular)
        df_regular.to_csv(output_file, index=False, encoding='utf-8-sig')


def main():
    tidy_all_seasons("get data", 2015, 2018)


if __name__ == "__main__":
    main()
