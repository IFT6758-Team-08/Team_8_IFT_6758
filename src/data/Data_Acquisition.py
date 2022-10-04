import requests
import os
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#Step 1: Create a function that takes a GAME_ID as input and returns the JSON response from the endpoint.
def get_game_data(game_id):
    url = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(game_id)
    response = requests.get(url,verify=False)
    return response.json()

#Step 2: Create a function that takes a season as input and returns a list of GAME_IDs for all regular season games in that season. You can use the following endpoint to get a list of all games in a season:
def get_game_ids(season):
    url = 'https://statsapi.web.nhl.com/api/v1/schedule?season={}&expand=schedule.teams,schedule.linescore'.format(season)
    response = requests.get(url,verify=False)
    game_ids = []
    for game in response.json()['dates']:
        game_ids.append(game['games'][0]['gamePk'])
    return game_ids

#Step 3: Create a function that takes a season as input and returns a list of GAME_IDs for all playoff games in that season. You can use the following endpoint to get a list of all games in a season:
def get_playoff_game_ids(season):
    url = 'https://statsapi.web.nhl.com/api/v1/schedule?season={}&expand=schedule.teams,schedule.linescore&leagueId=3'.format(season)
    response = requests.get(url,verify=False)
    game_ids = []
    for game in response.json()['dates']:
        game_ids.append(game['games'][0]['gamePk'])
    return game_ids

#Step 4: Create a function that takes a season as input and returns a list of GAME_IDs for all games in that season (regular season and playoffs).
def get_all_game_ids(season):
    return get_game_ids(season) + get_playoff_game_ids(season)

#Step 5: Create a function that takes a season as input and returns a list of all play-by-play data for all games in that season. You can use the get_game_data function you created in Step 1.
def get_all_game_data(season):
    game_ids = get_all_game_ids(season)
    game_data = []
    for game_id in game_ids:
        game_data.append(get_game_data(game_id))
    return game_data

def get_game_datas(year,filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    game_data = get_all_game_data(year)
    with open(filepath + '/{}.json'.format(year), 'w') as f:
        json.dump(game_data, f)
    return game_data    

#get all game data from 2016 to 2021
for year in range(2016,2022):
    get_game_datas(year,'./data/{}_game_data.json'.format(year))