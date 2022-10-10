# import requests
# import os
# import json
# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# #Step 1: Create a function that takes a GAME_ID as input and returns the JSON response from the endpoint.
# def get_game_data(game_id):
#     url = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(game_id)
#     response = requests.get(url,verify=False)
#     return response.json()

# #Step 2: Create a function that takes a season as input and returns a list of GAME_IDs for all regular season games in that season. You can use the following endpoint to get a list of all games in a season:
# def get_game_ids(season):
#     url = 'https://statsapi.web.nhl.com/api/v1/schedule?season={}&expand=schedule.teams,schedule.linescore'.format(season)
#     response = requests.get(url,verify=False)
#     game_ids = []
#     for game in response.json()['dates']:
#         game_ids.append(game['games'][0]['gamePk'])
#     return game_ids

# #Step 3: Create a function that takes a season as input and returns a list of GAME_IDs for all playoff games in that season. You can use the following endpoint to get a list of all games in a season:
# def get_playoff_game_ids(season):
#     url = 'https://statsapi.web.nhl.com/api/v1/schedule?season={}&expand=schedule.teams,schedule.linescore&leagueId=3'.format(season)
#     response = requests.get(url,verify=False)
#     game_ids = []
#     for game in response.json()['dates']:
#         game_ids.append(game['games'][0]['gamePk'])
#     return game_ids

# #Step 4: Create a function that takes a season as input and returns a list of GAME_IDs for all games in that season (regular season and playoffs).
# def get_all_game_ids(season):
#     return get_game_ids(season) + get_playoff_game_ids(season)

# #Step 5: Create a function that takes a season as input and returns a list of all play-by-play data for all games in that season. You can use the get_game_data function you created in Step 1.
# def get_all_game_data(season):
#     game_ids = get_all_game_ids(season)
#     game_data = []
#     for game_id in game_ids:
#         game_data.append(get_game_data(game_id))
#     return game_data

# def get_game_datas(year,filepath):
#     if not os.path.exists(filepath):
#         os.makedirs(filepath)
#     game_data = get_all_game_data(year)
#     with open(filepath + '/{}.json'.format(year), 'w') as f:
#         json.dump(game_data, f)
#     return game_data    

# #get all game data from 2016 to 2021
# for year in range(2016,2022):
#     get_game_datas(year,'./data/{}_game_data.json'.format(year))


import os, sys
import argparse
import requests
import json
import time

"""
The user passes as command line argument:
    The seasons to download: from_year - to_year (where to_year is optional) 

    A directory where the data files will be stored

"""


def get_data_regular(year, file_path):
    """
    The function takes as input year representing a season and a file path
    and downloads all the play_by_play events of the regular season
    in file_path. 



    """
    start = time.time()

    #if the file already exists, the function just loads it
    if os.path.isfile(file_path):
        with open(file_path, "r") as f_regular: 
            json.load(f_regular)
            return
    
    else:


        #iterating through all the games until we reach a game with no data (game with no 'liveData')
        
        all_regular_games = {} #dictionary (key, value) = (ID, game_data)
        
        game_exists = True #while the url contains information about the game
        i = 1 

        #This loop will go through all ids from 1 to the number of games  
        while game_exists == True:
            GAME_ID = "{}02{:04d}".format(year, i)
            url = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(GAME_ID)
            r = requests.get(url).json() #request all data about the game
            regular_game = {} 
            if r.get('liveData') is not None: #if 'liveData' == None, then no data for that game.
                regular_game['gameData'], regular_game['liveData'] = r['gameData'],r['liveData']
                all_regular_games[GAME_ID] = regular_game
                i +=1
            else:
                game_exists = False

        #write the json file: 
        with  open(file_path, 'w') as f_regular:
            json.dump(all_regular_games, f_regular)
        end = time.time()
        print("time for regular season", end - start)
        return 
 
#Playoffs
def get_data_playoffs(year, file_path):
    """ The function takes a year representing a season and a file path
    and downloads all the play_by_play events of the regular season
    in file_path.
    """
    
    start = time.time()

    #if the file has already been downloaded
    if os.path.isfile(file_path):
        with open(file_path, "r") as f_playoffs: 
            json.load(f_playoffs)
        return 

    
    else: 
        all_playoff_games = {} #stores the data for all games

        #This loop iterates through all the possible IDs of a playoff
        for p_round in range(1, 5): #4rounds
            for match_up in range(1, 2**(4-p_round)+1): #2^(4-round) matchups in each round 
                for game in range(1,8): #7 games for each matchup
                    GAME_ID = '{}030{}{}{}'.format(year, p_round, match_up, game)
                    url = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(GAME_ID)
                    r = requests.get(url).json()
                    playoff_game = {} 
                    if r.get('liveData') is not None: #if the game has no data / no play_to_play data
                        playoff_game['gameData'], playoff_game['liveData'] = r['gameData'],r['liveData']
                        all_playoff_games[GAME_ID] = playoff_game

        
        with open(file_path, 'w') as f_playoff:
            json.dump(all_playoff_games, f_playoff)
            
        end = time.time()
        print("time for playoff season", end - start)
        return 

def main():
    #This firt part (until the loop) can be removed if we don't need command line arguments" 
    #Have to specify: year, to_year and path to the folder where the files will be downloaded
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-y', '--year', type=int, required=True)
    # parser.add_argument('-p', '--path', type=str, required=True)
    # parser.add_argument('-ty', '--to_year', type=int)
    # args = parser.parse_args()

    # year, path = args.year, args.path
    # to_year = year 
    # if len(sys.argv) == 7:
    #     to_year = args.to_year
    year = 2016
    to_year = 2020
    path = "/home/yashaswi/Documents/IFT 6758/Project NHL"
     
    for y in range(year, to_year+1): 
        #file_path_regular, file_path_playoff = os.path.join(path, str(year) + "_regular_season.json"), os.path.join(path, str(year) + "_playoffs.json"),
        file_path_regular, file_path_playoff = path + "/" + str(y) + "_regular_season.json", path + "/" + str(y) + "_playoffs.json"
        get_data_playoffs(y, file_path_playoff) 
        get_data_regular(y, file_path_regular)



if __name__ == "__main__":
    main()
# Source Code: Lamia