import os, sys
import pandas as pd 
import numpy as np
import json 
from collections import Counter




def get_games_num(df): 
    """
    This function takes a dataframe with play_by_play events of the whole season 
    computes the number of games a specific team played in that season. 
    """ 
    list_teams =list(df['team'].unique()) 
    games_per_team = {}
    for team in list_teams: 
        new_df = df[df['team'] == team]
        games_per_team[team] = new_df['game_id'].unique().shape[0]
    print(games_per_team)
    return games_per_team



def aggregate_team_location(df, games_per_team): 
    """
    Computes the number of shots per location for each team
    And the shot rate per hour = total # of shots in a location/total number of games (since 1game = 1h)
    """
    df['y'] = df['y_transformed']+42.5
    #new_df = df.groupby(['team', 'y_transformed','goal_dist'])['event'].size().to_frame('total').reset_index()
    y_bins, goal_dist_bins = list(range(0,85,6)), list(range(-10,90,6))
    df['y_bins'], df['goal_dist_bins'] = pd.cut(df['y'], y_bins), pd.cut(df['goal_dist'], goal_dist_bins)
    new_df = df.groupby(['team', 'y_bins','goal_dist_bins'])['event'].size().to_frame('total').reset_index()
    new_df['games_per_team'] = new_df['team'].apply(lambda x: games_per_team.get(x))
    new_df['average_per_hour'] = new_df['total'] / new_df['games_per_team']
    new_df['y_mid'] = new_df['y_bins'].apply(lambda x: (x.left + x.right)/2)
    new_df['goal_mid'] = new_df['goal_dist_bins'].apply(lambda x: (x.left + x.right)/2)
    new_df.to_csv("./2016_team.csv", index = False, encoding='utf-8-sig')
    return new_df

def aggregate_shot_location(df): 
    """
    Computes the number of shots at each location for the whole league 
    And the shot rate per hour = total # of shots in a location/total number of games (since 1game = 1h)
    """
    df['y'] = df['y_transformed']+42.5
    num_teams = len(df['team'].unique())
    total_games = df['game_id'].unique().shape[0]

    y_bins, goal_dist_bins = list(range(0,85,6)), list(range(-10,90,6))
    df['y_bins'], df['goal_dist_bins'] = pd.cut(df['y'], y_bins), pd.cut(df['goal_dist'], goal_dist_bins)
    new_df = df.groupby(['y_bins','goal_dist_bins'])['event'].size().to_frame('total').reset_index()
    new_df['average_per_hour'] = new_df['total']/total_games
    new_df['y_mid'] = new_df['y_bins'].apply(lambda x: (x.left + x.right)/2)
    new_df['goal_mid'] = new_df['goal_dist_bins'].apply(lambda x: (x.left + x.right)/2)
    new_df.to_csv("./2016_league.csv", index = False, encoding='utf-8-sig')
    return new_df

    
def transform_coordinates(rinkSide, coor): 
    """
    This function transforms a coordinate from the left offensive side to the right one. 
    """ 
    if rinkSide == "right": #the offensive zone is left --> transformation
        return (-1)*coor
    else: #if left or nothing
        return coor


#function to normalize coordinates and compute distances
def add_transformed_col(df): 
    """
    This function adds columns to the data frame: 
     - x_transformed and y_transformed: the coordinates transposed to the right side of the rink 
     - euclidean distance: the euclidean distance from a shot location to the center of the rink 
     - goal_dist: the distance from a shot location to the goal line: 89-x_transformed 
     - y_dist: if the shot was above or below the center rink. 
    """
    df['x_transformed'] = df.apply(lambda x:transform_coordinates(x['rinkSide'], x['coordinates_x']), axis=1)
    df['y_transformed'] = df.apply(lambda x:transform_coordinates(x['rinkSide'], x['coordinates_y']), axis=1)
    df = df.drop(df[df.x_transformed < 25].index) #if <25 then not in the offensive zone. 
    df['goal_dist'] = df.apply(lambda x: (89 - x['x_transformed']), axis = 1)
    return df


def get_season_agg(y_mid, goal_mid, league_df):
    """"
    This function returns the shot rate per hour of the whole league for a specific location. 
    """
    league = league_df.loc[(league_df["y_mid"]==y_mid) & (league_df["goal_mid"]== goal_mid), 'average_per_hour']
    return league.iloc[0]







def main(): 
    df = pd.read_csv("./2016_clean.csv")
    df = df.dropna(subset=['rinkSide']) #drop the events where rinkide is not present. 
    df = add_transformed_col(df) #add columns: x,y transposed to the right side,  goal dist, and on which side of the y_axis (up or down)
   
    #df.to_csv("./2016_transformed.csv", index = False, encoding='utf-8-sig')
    games_per_team = get_games_num(df) #dictionary key = team name, value=number of games played in the whole season. 
    df_league = league_df = aggregate_shot_location(df) #df with shot rate/h grouped by location (across all teams)
    df_team = aggregate_team_location(df, games_per_team) #df with shot rate/h grouped by team and location

    #league_full_rink = get_full_wink_av(league_df)
    #Add the corresponding shot rate/h of the league in each row of df_team. 
    df_team['league_avearage'] = df_team.apply(lambda x: get_season_agg(x['y_mid'], x['goal_mid'], df_league), axis=1)
    df_team['raw_diff'] = df_team['average_per_hour'] - df_team['league_avearage']
    df_team.to_csv("./2016_diff.csv", index = False, encoding='utf-8-sig')
    #df_team['diff_percent'] = 100*2*np.abs(df_team['average_per_hour'] - df_team['league_avearage']) / (df_team['average_per_hour'] + df_team['league_avearage'])
    #df_team['percent_normalized'] = 2*((df_team['diff_percent'] - df_team['diff_percent'].min()) / (df_team['diff_percent'].max() - df_team['diff_percent'].min()))-1
    #df_team['raw_normalized'] = 2*((df_team['raw_diff'] - df_team['raw_diff'].min()) / (df_team['raw_diff'].max() - df_team['raw_diff'].min()))-1
    #df_team.to_csv("./2016_diff.csv", index = False, encoding='utf-8-sig')

if __name__ == "__main__": 
    main()