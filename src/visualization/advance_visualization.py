import os, sys
import pandas as pd 
import numpy as np
from scipy.ndimage import gaussian_filter


def transform_coordinates(season_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds columns to the dataframe: 
     - x_transformed and y_transformed: the coordinates transposed to the right side of the rink 
     - goal_dist: the distance between a shot location and goal line
    """
    #Represent the coordinates on the right side of the rink 
    season_df['x_transformed'] = season_df.apply(lambda x:transform_one_coordinate(x['rinkSide'], x['coordinates_x']), axis=1)
    season_df['y_transformed'] = season_df.apply(lambda x:transform_one_coordinate(x['rinkSide'], x['coordinates_y']), axis=1)
   
    #Remove the shots that were not in the offisive zone (x<25) and the ones made after the goal line
    season_df = season_df.drop(season_df[(season_df.x_transformed < 25) & (season_df.x_transformed > 89)].index)  

    #computes the distace from the goal (dist = 89 - x)
    season_df['goal_dist'] = season_df.apply(lambda x: (89 - x['x_transformed']), axis = 1)
    return season_df

def transform_one_coordinate(rinkSide, coor) : 
    """
    This function transforms a coordinate from the left offensive side to the right one. 
    """ 
    #if a team is playing of the right side, their offensive zone is on the left and coor is changed: 
    if rinkSide == "right": 
        return (-1)*coor
    else: 
        return coor



def compute_league_rate(season_df) : 
    """
    Computes the number of shots at binned location for the whole league 
    and the league shot rate per hour
    """
    #total number of games in a season for the whole league: 
    total_game_num = season_df['game_id'].unique().shape[0] 

    #binning the coordinates and grouping by bined location: 
    season_df['y'] = season_df['y_transformed']*(-1) #to fit the rink representation in the plot. 
    y_bins, goal_dist_bins = list(range(-41,42,4)), list(range(0,94,4))
    season_df['y_bins'], season_df['goal_dist_bins'] = pd.cut(season_df['y'], y_bins), pd.cut(season_df['goal_dist'], goal_dist_bins)
    new_df = season_df.groupby(['y_bins','goal_dist_bins'])['event'].size().to_frame('total').reset_index()

    #midpoints of the bins: 
    new_df['y_mid'] = new_df['y_bins'].apply(lambda x: (x.left + x.right)/2)
    new_df['goal_mid'] = new_df['goal_dist_bins'].apply(lambda x: (x.left + x.right)/2)

    #Average shot rate per hour for the whole league (there are 2 teams during each game)
    new_df['league_rate'] = new_df['total']/(2*total_game_num) 

    return new_df


def compute_team_rate(season_df): 
    """
    Computes the number of shots per location for each team
    And the shot rate per hour 
    """

    #Dictionary (key, value) = (team, total num of games played in the season)
    num_games_per_team = get_games_num(season_df)
    
    #binning the coordinates and grouping by bined team and location: 
    season_df['y'] = season_df['y_transformed']*(-1) #to fit the rink representation in the plot. 
    y_bins, goal_dist_bins = list(range(-41,42,4)), list(range(0,94,4))
    season_df['y_bins'], season_df['goal_dist_bins'] = pd.cut(season_df['y'], y_bins), pd.cut(season_df['goal_dist'], goal_dist_bins)
    new_df = season_df.groupby(['team', 'y_bins','goal_dist_bins'])['event'].size().to_frame('total').reset_index()



    #Midpoints of the bins: 
    new_df['y_mid'] = new_df['y_bins'].apply(lambda x: (x.left + x.right)/2)
    new_df['goal_mid'] = new_df['goal_dist_bins'].apply(lambda x: (x.left + x.right)/2)

    #Computing the average shot rate per team: 
    new_df['total_games'] = new_df['team'].apply(lambda x: num_games_per_team.get(x))
    new_df['team_rate'] = new_df['total'] / new_df['total_games']

    #new_df.to_csv("./201_team.csv", index = False, encoding='utf-8-sig')
    return new_df

def get_games_num(df): 
    """
    This function takes a dataframe with play_by_play events of the whole season and 
    returns a dictionary: (key, value) = (team, total number of games played in a season)
    """ 
    list_teams =list(df['team'].unique()) 
    games_per_team = {}
    for team in list_teams: 
        new_df = df[df['team'] == team]
        games_per_team[team] = new_df['game_id'].unique().shape[0]
    return games_per_team

def get_one_loc_league(y_mid, goal_mid, league_df):
    """"
    This function returns the shot rate per hour of the whole league for a specific location. 
    """

    league = league_df.loc[(league_df["y_mid"]==y_mid) & (league_df["goal_mid"]== goal_mid), 'league_rate']
    return league.iloc[0]


def compute_diff(df_league, df_team): 

    #Adding a column to df_team for league shot average at each location 
    df_team['league_rate'] = df_team.apply(lambda x: get_one_loc_league(x['y_mid'], x['goal_mid'], df_league), axis=1)
    df_team['raw_diff'] = df_team['team_rate'] - df_team['league_rate']

    #Computing the difference in each location
    df_team['raw_diff'] = df_team['team_rate'] - df_team['league_rate']
    return df_team

def main(): 
    season_df = pd.read_csv("2016_clean.csv").dropna(subset=['rinkSide']) 
    season_df = transform_coordinates(season_df) # transforms the coordinates to the right side of the rink

    df_league = compute_league_rate(season_df) #df with shot rate/h grouped by location (across all teams)
    df_team = compute_team_rate(season_df) #df with shot rate/h grouped by team and location 

    #Compute the excess shot rate at each location
    rate_df = compute_diff(df_league, df_team)

    rate_df.to_csv("2016_difference.csv", index = False, encoding='utf-8-sig')      


if __name__ == "__main__": 
    main()