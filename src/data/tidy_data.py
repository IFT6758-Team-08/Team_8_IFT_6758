import os, sys
import pandas as pd
import json



def tidy_data(data_dict: dict):
    #for (game_id, game_data) in data_dict.items():
        #events_list = game_data['liveData']['plays']['all_plays']
    pass 
        
    
def main():
    df = pd.read_json('./test.json')
    print(df)


if __name__ == "__main__":
    main() 
