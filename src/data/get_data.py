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
        full_regular_data = [] #stores data of each game
        game_exists = True
        i = 1

        #This loop will go through all ids from 1 to the number of games  
        while game_exists == True:
            GAME_ID = "{}02{:04d}".format(year, i)
            url = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(GAME_ID)
            r = requests.get(url).json() #request all data about the game
            if r.get('liveData') is not None: #if 'liveData' == None, then no data for that game. 
                regular_season = r['liveData']['plays']['allPlays'] #but only save the play_by_play part we need
                full_regular_data.append(regular_season)
                i +=1
            else:
                game_exists = False

        #write the json file: 
        with  open(file_path, 'w') as f_regular:
            json.dump(full_regular_data, f_regular)
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
        full_playoff_data = [] #stores the data for all games

        #This loop iterates through all the possible IDs of a playoff
        for p_round in range(1, 5): #4rounds
            for match_up in range(1, 2**(4-p_round)+1): #2^(4-round) matchups in each round 
                for game in range(1,8): #7 games for each matchup
                    GAME_ID = '{}030{}{}{}'.format(year, p_round, match_up, game)
                    url = 'https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/'.format(GAME_ID)
                    r = requests.get(url).json()
                    if r.get('liveData') is not None: #if the game has no data / no play_to_play data
                        playoffs = r['liveData']['plays']['allPlays']
                        full_playoff_data.append(playoffs)

        
        with open(file_path, 'w') as f_playoff:
            json.dump(full_playoff_data, f_playoff)
            
        end = time.time()
        print("time for playoff season", end - start)
        return 

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', type=int, required=True)
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-ty', '--to_year', type=int)
    args = parser.parse_args()

    year, path = args.year, args.path
    print(len(sys.argv))
    if len(sys.argv) == 7:
        to_year = args.to_year
    else: to_year= year  

    for y in range(year, to_year+1): 
        #file_path_regular, file_path_playoff = os.path.join(path, str(year) + "_regular_season.json"), os.path.join(path, str(year) + "_playoffs.json"),
        file_path_regular, file_path_playoff = path + "/" + str(y) + "_regular_season.json", path + "/" + str(y) + "_playoffs.json"
        get_data_playoffs(y, file_path_playoff) 
        get_data_regular(y, file_path_regular)



if __name__ == "__main__":
    main()
