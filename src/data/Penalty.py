import numpy as np
import pandas as pd
import json


def get_penalty_plays(path):
    """returns a dictionary with list of penalty plays as values and game_id as keys."""
    data = pd.read_csv(path)
    data = data[np.floor(data['game_id']/1000000) == 2016]
    game_ids = data['game_id']
    game_ids = np.unique(game_ids)

    all_penalty_plays = {}
    for game_id in game_ids:
        value = data[data['game_id'] == game_id]
        value = value[value['event'] == 'Penalty']
        all_penalty_plays[game_id] = list(value.index)  #penalty plays are the index of the penalty event in the dataframe
    return all_penalty_plays


def calc_time(pperiod, pstart_time):
    t = pstart_time.split(":")
    m = int(t[0])
    s = int(t[1])
    sec = m * 60 + s

    if pperiod < 5:
        time = (pperiod - 1) * 20 * 60 + sec
    else:
        time = (4 * 20 * 60) + (5 * 60)
    return time

def penalty_occured(powerplay_start_time, t, stack, delayed, row):
    minutes = row['penalty_minutes']
    pstart_time = row['period_time']
    pperiod = int(row['period'])
    current_time = calc_time(pperiod, pstart_time)
    player_numbers = row[t + '_number_of_players']
    if t == 'away':
        other_team_players = row['home_number_of_players']
    else:
        other_team_players = row['away_number_of_players']

    if player_numbers > 3:
        row[t + '_number_of_players'] = player_numbers - 1
        stack.append((minutes, current_time, player_numbers - 1))
    else: #we should have delayed penalty
        delayed.append((minutes, current_time, player_numbers))

    if row[t + '_number_of_players'] != other_team_players:  #it is power play
        if powerplay_start_time == 0: #we werent already on a powerplay
            powerplay_start_time = current_time

    else:
        row['time_since_power_play'] = 0 #reset power play
        powerplay_start_time = 0

    if powerplay_start_time != 0:
        row['time_since_power_play'] = current_time - powerplay_start_time

    return powerplay_start_time, stack, delayed, row


def check_reset_penalty_time(home_penalties, away_penalties, row, powerplay_start_time):
    current_time = calc_time(row['period'], row['period_time'])
    new_home_penalties = home_penalties.copy()
    new_away_penalties = away_penalties.copy()
    new_row = row.copy()
    i = 0
    reset1, reset2 = False, False
    for penalty in home_penalties:
        #(minutes, current_penalty_time, player_numbers)
        if penalty[1] + penalty[0] * 60 <= current_time:
            #we should reset this penalty
            print("reset this penalty1")
            print(row)
            reset1 = True
            del new_home_penalties[i]
            new_row['home_number_of_players'] += 1
        i += 1
    i = 0
    for penalty in away_penalties:
        #(minutes, current_penalty_time, player_numbers)
        if penalty[1] + penalty[0] * 60 <= current_time:
            print("reset this penalty2")
            print(row)
            #we should reset this penalty
            reset2 = True
            del new_away_penalties[i]
            new_row['away_number_of_players'] += 1
        i+=1


    if new_row['home_number_of_players'] == new_row['away_number_of_players']: #even strength
        if home_delayed == [] and away_delayed == []:  ###################
            # powerplay ends
            new_row['time_since_power_play'] = 0
            powerplay_start_time = 0
    else:
        if powerplay_start_time == 0: #not already on a powerplay
            powerplay_start_time = current_time

    if powerplay_start_time != 0:
        new_row['time_since_power_play'] = current_time - powerplay_start_time

    return new_home_penalties, new_away_penalties, new_row, powerplay_start_time, reset1, reset2


def unstack_delayed_penalty(stack, delayed, row, t):
    # (minutes, current_penalty_time, player_numbers)
    current_time = calc_time(row['period'], row['period_time'])
    if delayed != []:
        number_of_unstack = row[t + '_number_of_players'] - 3
        to_unstack = min(len(delayed), number_of_unstack) #number of penalties we can unstack now
        for p in range(to_unstack):
            penalty = delayed[0]
            del delayed[0]
            updated_penalty = (penalty[0], current_time, penalty[2])
            stack.append(updated_penalty)
        row[t + '_number_of_players'] = row[t + '_number_of_players'] - to_unstack

    return stack, delayed, row


def check_reset_penalty_goal(stack, row, powerplay_start_time, t):
    # stack = [(minutes, current_time, player_numbers), ...]
    print("its a goal")
    print(stack)
    current_time = calc_time(row['period'], row['period_time'])
    new_row = row.copy()
    i = 0
    reset = False
    for penalty in stack:  # only one penalty will cancel(in the order of occurance)
        minutes = penalty[0]
        penalty_start_time = penalty[1]
        if minutes == 2: #minor penalty
            #cancel this penalty
            del stack[i]
            new_row[t + '_number_of_players'] = row[t + '_number_of_players'] + 1
            reset = True
            break
        elif minutes == 4: #double minor penalty
            #cancle one minor penalty
            if current_time - penalty_start_time > 120: #less than 2 mins left
                del stack[i]
                new_row[t + '_number_of_players'] = row[t + '_number_of_players'] + 1
                reset = True
            else: #more than 2 mins left
                stack[i] = (2, stack[i][1], stack[i][2])
            break
        i += 1


    if new_row['home_number_of_players'] == new_row['away_number_of_players']: #even strength
        # powerplay ends
        new_row['time_since_power_play'] = 0
        powerplay_start_time = 0
    else:
        if powerplay_start_time == 0: #not already on a powerplay
            powerplay_start_time = current_time

    if powerplay_start_time != 0:
        new_row['time_since_power_play'] = current_time - powerplay_start_time

    return stack, new_row, powerplay_start_time, reset


def add_columns(df):
    df['time_since_power_play'] = 0
    df['away_number_of_players'] = 5
    df['home_number_of_players'] = 5
    df.loc[df['period'] == 4, 'home_number_of_players'] = 3
    df.loc[df['period'] == 4, 'away_number_of_players'] = 3
    return df

def get_teams(game_id):
    #specify home and away teams
    with open('get data/2016_regular_season.json') as file:
        data = json.load(file)
    teams = data[game_id]['gameData']['teams']
    away = teams['away']['name']
    home = teams['home']['name']
    return home, away

if __name__ == '__main__':
    all_penalty_plays = get_penalty_plays('M2_regular_bonus_2016_cleaned.csv')
    df = pd.read_csv('M2_regular_bonus_2016_cleaned.csv')
    df = pd.read_csv('bonus_test.csv')
    # df = df.iloc[375:405]
    df = add_columns(df)
    # print(df['event'])
    print(df.iloc[:, 8:])
    game_ids = all_penalty_plays.keys()

    game_ids = list(game_ids)[0:1]

    for game_id in game_ids:
        game_penalty_plays = all_penalty_plays[game_id]
        game_penalty_plays = [11, 13, 16, 18]
        home, away = get_teams(str(game_id))
        home_penalties = []
        away_penalties = []
        home_delayed = []
        away_delayed = []
        reset1, reset2, reset3, reset4 = False, False, False, False
        powerplay_start_time = 0
        if game_penalty_plays != []:
            last_idx = df[df['game_id'] == game_id].index[-1]

            # print(last_idx, game_penalty_plays, game_id)
            for play in range(game_penalty_plays[0], last_idx + 1):
                # play -= 375
                pteam = df.iloc[play]['team']
                #update number of players
                #play + 375
                df.at[play, 'away_number_of_players'] = df.iloc[play - 1]['away_number_of_players']
                df.at[play, 'home_number_of_players'] = df.iloc[play - 1]['home_number_of_players']

                #reset1 :home, reset2: away
                home_penalties, away_penalties, new_row, powerplay_start_time, reset1, reset2 = check_reset_penalty_time(home_penalties, away_penalties, df.iloc[play], powerplay_start_time)
                df.iloc[play] = list(new_row)

                if df.iloc[play]['event'] == 'Goal':
                    if pteam == home: #if home team scores goal, a penalty of away team should be canceled
                        if away_penalties != []:
                            away_penalties, new_row, powerplay_start_time, reset4 = check_reset_penalty_goal(away_penalties, df.iloc[play], powerplay_start_time, 'away')

                    else:
                        if home_penalties != []:
                            home_penalties, new_row, powerplay_start_time, reset3 = check_reset_penalty_goal(home_penalties, df.iloc[play], powerplay_start_time, 'home')

                    df.iloc[play] = list(new_row)
                #reset3: home, reset4: away
                #check if there was a reset in penalty, unstack delayed penalties
                if reset1 or reset3: #home penalty reset
                    home_penalties, home_delayed, new_row = unstack_delayed_penalty(home_penalties, home_delayed, df.iloc[play], 'home')
                    df.iloc[play] = list(new_row)
                if reset3 or reset4: #away penalty reset
                    away_penalties, away_delayed, new_row = unstack_delayed_penalty(away_penalties, away_delayed, df.iloc[play], 'away')
                    df.iloc[play] = list(new_row)

                if reset1 or reset2 or reset3 or reset4:  #if there was a reset, check for any strength change
                    current_time = calc_time(df.iloc[play]['period'], df.iloc[play]['period_time'])
                    if df.iloc[play]['home_number_of_players'] == df.iloc[play]['away_number_of_players']:  # even strength
                        # powerplay ends
                        df.at[play, 'time_since_power_play'] = 0
                        powerplay_start_time = 0
                    else:
                        if powerplay_start_time == 0:  # not already on a powerplay
                            powerplay_start_time = current_time

                    if powerplay_start_time != 0:
                        df.at[play, 'time_since_power_play'] = current_time - powerplay_start_time

                #play + 375
                if play in game_penalty_plays: #penalty occured
                    print(play)
                    minutes = df.iloc[play]['penalty_minutes']
                    if int(minutes) > 0:
                        if pteam == home:
                            powerplay_start_time, home_penalties, home_delayed, new_row = penalty_occured(powerplay_start_time, 'home', home_penalties, home_delayed, df.iloc[play])
                        else:
                           powerplay_start_time, away_penalties, away_delayed, new_row = penalty_occured(powerplay_start_time, 'away', away_penalties, away_delayed, df.iloc[play])

                        df.iloc[play] = list(new_row)

                reset1, reset2, reset3, reset4 = False, False, False, False
    print(df.iloc[:, 9:])

# a = get_penalty_plays('M2_regular_bonus_2016_cleaned.csv')
# add_columns()
# df = pd.read_csv('M2_regular_bonus_2016_cleaned.csv')
# print(df.iloc[70:80])