# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:44:27 2022

@author: Divya
"""

import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


def no_of_events(df_game_events): 
    df_shot_goal = df_game_events.groupby(['event','secondary_type']).size().reset_index().pivot(columns='event', index='secondary_type', values=0)
    df_shot_goal = df_shot_goal.reindex(columns=['Shot','Goal'])
    df_shot_goal.sort_values(['Shot','Goal'], ascending = [True, True], inplace=True)

    df_shot_goal.plot(kind='bar', stacked=True)
    plt.xlabel('Shot type')
    plt.ylabel('Number of shot/goal')

    df_total_shot_goal = df_shot_goal['Shot']+df_shot_goal['Goal']
    df_shot_goal['Total Shots'] = df_total_shot_goal
    df_shot_goal['Success Rate'] = df_shot_goal['Goal']/df_total_shot_goal
    df_shot_goal['goal_percent'] = (df_shot_goal["Goal"]/df_shot_goal["Goal"].sum()) *100
    print(df_shot_goal)
    return


def prep_data(path1, path2, path3):  
    df_2018 = pd.read_csv(path1)
    df_2019 = pd.read_csv(path2)
    df_2020 = pd.read_csv(path3)
    df_2018["season"] = "2018"
    df_2019["season"] = "2019"
    df_2020["season"] = "2020"
    df_all_season = pd.concat([df_2018,df_2019,df_2020]).reset_index(drop=True)
    df_all_season = df_all_season[df_all_season.rinkSide.notna() & df_all_season.coordinates_x.notna() & df_all_season.coordinates_y.notna()].reset_index(drop=True)
    df_all_season["offensive_goal_post_x"] = [-89 if i=="right" else 89 for i in df_all_season["rinkSide"]]
    df_all_season["offensive_goal_post_y"] = 0
    df_all_season["shot_distance"] = df_all_season.apply(lambda x: np.sqrt((x['offensive_goal_post_x']-x['coordinates_x'])**2 + (x['offensive_goal_post_y']-x['coordinates_y'])**2), axis=1)
    df_all_season['goal'] = np.where(df_all_season['event']=="Goal", True, False)
    return df_all_season


def distance_vs_goal(df_game_events):
    df_game_events['shot_distance'] = df_game_events['shot_distance'].round(0)
    fig = plt.figure(figsize=(15, 20))
    for season_idx, season in enumerate(['2018','2019','2020']):
        plt.subplot(3, 1, season_idx + 1)
        df_game_events_season = df_game_events[df_game_events['season'].astype(str)==season]
        df_game_events_season = df_game_events_season.groupby(['shot_distance'])['goal'].mean().to_frame().reset_index()
        ax = sns.lineplot(x='shot_distance', y='goal', data=df_game_events_season)
        plt.title(f'Season {season}')
        ax.set_axisbelow(True)
        plt.xticks(np.arange(0, 200, 5), rotation=90)
        plt.xlabel('Shot distance (ft)')
        plt.ylabel('Goal Probability')
        
    plt.suptitle('Relationship between shot distance and chance of goal \n for seasons 2018, 2019, 2020', size=12, y=0.92)


def shot_type_vs_goal_percent(df_game_events):
    df_game_events = df_game_events[df_game_events.rinkSide.notna() & df_game_events.coordinates_x.notna() & df_game_events.coordinates_y.notna()].reset_index(drop=True)
    df_game_events["offensive_goal_post_x"] = [-89 if i=="right" else 89 for i in df_game_events["rinkSide"]]
    df_game_events["offensive_goal_post_y"] = 0
    df_game_events["shot_distance"] = df_game_events.apply(lambda x: np.sqrt((x['offensive_goal_post_x']-x['coordinates_x'])**2 + (x['offensive_goal_post_y']-x['coordinates_y'])**2), axis=1)
    df_game_events['goal'] = np.where(df_game_events['event']=="Goal", True, False)

    bins = pd.cut(df_game_events['shot_distance'], 15)

    df_goal = df_game_events.groupby([bins,'secondary_type'])['event'].apply(lambda x: (x=='Goal').sum())
    df_shot = df_game_events.groupby([bins, 'secondary_type'])['event'].apply(lambda x: (x=='Shot').sum())
    df_goal_shot = pd.merge(df_goal, df_shot, how='inner',left_index=True, right_index=True).reset_index().rename(columns={'event_x':'n_goal', 'event_y':'n_shot', 'shot_distance':'binned_distance'})
    df_goal_shot['percentage_goal'] = df_goal_shot['n_goal']/(df_goal_shot['n_goal']+df_goal_shot['n_shot']) * 100
    df_goal_shot['binned_distance'] = df_goal_shot.binned_distance.apply(str)
    df_final = df_goal_shot.pivot_table(values='percentage_goal', index=['binned_distance'], columns=['secondary_type']).reset_index()
    df = df_final.set_index("binned_distance")
    df = df.fillna(0)
    
    plt.figure()
    plt.rcParams.update({'font.size': 24})
    ax = df.plot(kind='bar', rot=90, width=0.8, fontsize=20, xlabel='Binned Distance', ylabel='Goal Percentage', figsize=(30, 10))
    ax.set_title("Goal Percentage for all Shot Types", pad=20, fontdict={'fontsize':24})
    
    
df_game_events = pd.read_csv('2018_clean.csv')
no_of_events(df_game_events)

df_all_season = prep_data('2018_clean.csv', '2019_clean.csv', '2020_clean.csv')      
distance_vs_goal(df_all_season)

shot_type_vs_goal_percent(df_game_events)

