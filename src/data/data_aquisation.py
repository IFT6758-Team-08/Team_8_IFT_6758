import time
import requests
import json
import os

"""
references:
Download_Data: https://www.askpython.com/python/examples/pull-data-from-an-api 
"""


def download_seasonal_data(season, path, game_type):
    """download data of a whole season"""

    if game_type == 'regular_season': # initialize
        prev_game_number = '0000'
    elif game_type == 'playoff':
        prev_game_number = '0110'
    prev_game_available = True
    game_id = '0000000000'

    while (game_id != 'end'):
        game_id = seasonal_game_id(season, game_type, prev_game_number, prev_game_available)
        prev_game_available = True
        # print(game_id)
        url = 'https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/'.format(game_id=game_id)
        parse_json = download_data(url)
        if (len(parse_json.keys()) == 2):  # url with the given game_id not available
            prev_game_available = False

        else:
            filtered_json = parse_json
            write_to_file(filtered_json, game_id, path)
        prev_game_number = game_id[6:]


def download_data(url):
    """Download the raw data from the link passed as arg"""

    response_api = requests.get(url)
    data = response_api.text
    parse_json = json.loads(data)
    # print(parse_json.keys())

    return parse_json


def seasonal_game_id(season, game_type, prev_game_number, prev_game_available):
    """returns next possible game id of a chosen season"""

    playoff_id = '03'
    regual_season_id = '02'

    if game_type == 'regular_season':
        game_type_id = regual_season_id
        game_number = str(int(prev_game_number) + 1)
        game_number = "0"*(4-len(game_number)) + game_number
        if prev_game_available == False:
            game_id = 'end'
            return game_id

    elif game_type == 'playoff':
        game_type_id = playoff_id
        new_game_number_status, game_number = new_playoff_game_number(prev_game_number, prev_game_available)
        if new_game_number_status == 'end':
            return new_game_number_status

    game_id = season + game_type_id + game_number
    # print(game_id)
    return game_id


def new_playoff_game_number(prev_game_number, prev_game_available):
    """returns the last 4 digits of a playoff game id, according to the last playoff id used to download"""

    new_game_number = prev_game_number
    # print(prev_game_available)
    if prev_game_available:
        if prev_game_number[3] == '7': # last game. we should go to the next matchup
            new_game_number = change_string(new_game_number, 2, 4, str(int(new_game_number[2:]) + 4)) # increase the matchup and set the game to 1
            new_game_number_status, new_game_number = check_new_game_number(new_game_number)
        else: #not the last game. We should go to the next game
            new_game_number = change_string(new_game_number, 3, False, str(int(new_game_number[3]) + 1))
            # print(new_game_number)
            new_game_number_status = 'continue'
    else: #if prev number was unavailable
        new_game_number = change_string(new_game_number, 2, 4, str(int(new_game_number[2:]) + 10)) # So we should go on to the next matchup
        new_game_number = change_string(new_game_number, 3, False, '1')
        new_game_number_status, new_game_number = check_new_game_number(new_game_number)
        print("im here")
    return new_game_number_status, new_game_number


def check_new_game_number(new_game_number):
    # print(new_game_number[1:3])
    if new_game_number[1:3] == '42':  # check the 2nd and 3rd digits of the last 4 digits
        new_game_number_status = 'end'  # end of last(4th) round
    elif new_game_number[1:3] == '33':  # end of 3rd round
        new_game_number_status = 'continue'
        new_game_number = change_string(new_game_number, 1, 4, '411')
    elif new_game_number[1:3] == '25':  # end of 2nd round
        # print("yea")
        new_game_number_status = 'continue'
        new_game_number = change_string(new_game_number, 1, 4, '311')
    elif new_game_number[1:3] == '19':  # end of 1st round
        new_game_number_status = 'continue'
        new_game_number = change_string(new_game_number, 1, 4, '211')
    else:
        new_game_number_status = 'continue'
    return new_game_number_status, new_game_number


def change_string(string, index_start, index_end, new_char):
    string_list = list(string)
    # new_char_list = list(new_char)
    if index_end == False:
        string_list[index_start] = new_char
    else:
        string_list[index_start:index_end] = new_char
    new_string = "".join(string_list)
    # print(new_string)
    return new_string


def write_to_file(parse_json, game_id, path):
    """writes a dictionary(json data) to a json file"""

    if not os.path.exists(path): #create path
        os.makedirs(path)
    path = path + game_id + ".json"
    if not os.path.exists(path):
        with open(path, "w") as outfile:
            json.dump(parse_json, outfile)
    print(game_id)


start = time.time()
download_seasonal_data('2020', 'raw_data/2020/', 'playoff')
end = time.time()
print("time is: ", end - start, " (s)")
