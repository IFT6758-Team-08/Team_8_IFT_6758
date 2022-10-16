  
import os, sys
import pandas as pd
import json

#2016020007: 5 periods but only 4 in linescore

def get_game_events(data: dict):
    """
        This function takes a dictionary with a game's information (gameData and liveData)
        and returns two dictionaries: 
            games_metadata: key=game_ID, values = dictionary with the rinkSide of each team for every period
            events_dict: key=game_ID, value =  all play by play events
    """
    events_dict, games_metadata = {}, {}
    for (key, info) in data.items():
        game_meta, events_list = {}, []
        period_info_list = info['liveData']['linescore']['periods'] #list containing the team's side in each period
        if len(period_info_list) >0:
            game_meta[info['gameData']['teams']['home']['name']], game_meta[info['gameData']['teams']['away']['name']] = get_metadata_rinkSide(period_info_list)
        for event in info['liveData']['plays']['allPlays']: 
            if event['result']['event'] == "Shot" or event['result']['event'] == "Goal": #filters all other events and games with no data
                events_list.append(event)
        games_metadata[key] = game_meta
        events_dict[key] = events_list
    return events_dict, games_metadata

  
def get_metadata_rinkSide(period_info_list): 
    game_period_home, game_period_away = {} , {}
    for i, period in enumerate(period_info_list):  #ex. 2017020045 no rinkside
        if 'rinkSide' in period['home']: 
            game_period_home[str(i+1)] = period['home']['rinkSide']
            game_period_away[str(i+1)] = period['away']['rinkSide']
    return game_period_home, game_period_away

  
def filter_df(df, games_metadata):
    """
        This function filters a dataframe and only keeps a subset of columns
    """
    new_df = pd.DataFrame()
    new_df[['game_id', 'period', 'period_time',
                 'team', 'event', 'coordinates_x',
                 'coordinates_y', 'secondary_type',
                 'empty_net', 'strength_name']] = df[['game_id', 'about.period', 'about.periodTime',
                 'team.name', 'result.event', 'coordinates.x',
                 'coordinates.y', 'result.secondaryType',
                 'result.emptyNet', 'result.strength.name']]
    new_df_copy = new_df.copy()
    new_df_copy['rinkSide'] = new_df.apply(lambda x:get_rinkside(x['game_id'], x['team'], x['period'], games_metadata), axis=1)
    new_df_copy['shooter'], new_df_copy['goalie'] = df['players'].apply(get_shooter), df['players'].apply(get_goalie)
    return new_df_copy

  
def get_rinkside(game_id, team, period, games_metadata): 
  """
    This function returns the rinkSide of a team depending of the period 
  """
    if str(period) in games_metadata[game_id][team]: 
        rinkside = games_metadata[game_id][team][str(period)]
    else: 
        rinkside = None
    return rinkside

  
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

    with open(path) as file:
        data = json.load(file)
    filtered_dict, games_metadata = get_game_events(data)  # dictionary with 2 keys: game_id and play_by_play events
    game_ids = filtered_dict.keys()
    all_df = pd.DataFrame()
    for game_id in game_ids:
        df = pd.json_normalize(filtered_dict[game_id])  # flattens the embedded dictionaries
        df['game_id'] = game_id
        all_df = pd.concat([all_df, df])
    all_df.reset_index(inplace=True, drop=True)

    #shots_goals_df = get_shots_goals_events(all_df)  # only keeps the shots and goals events
    filtered_df = filter_df(all_df,games_metadata)  # keeps only the columns we need
    return filtered_df

  
def tidy_all_seasons(path, year, to_year):
    """
    returns the dataframe for all the seasons(from year to to_year)
    """
    for y in range(year, to_year + 1):
       # all_df = pd.DataFrame()
        file_path_regular, file_path_playoff = path + "/" + str(y) + "_regular_season.json", path + "/" + str(y) + "_playoffs.json"
        output_file = path + "/" + str(y) +"_clean.csv"
        df_regular = tidy_one_season(file_path_regular)
        df_playoff = tidy_one_season(file_path_playoff)
        all_df = pd.concat([df_regular, df_playoff])
        all_df.to_csv(output_file, index = False, encoding='utf-8-sig')


def main():
    tidy_all_seasons(".", 2016,2020)



if __name__ == "__main__":
    main()