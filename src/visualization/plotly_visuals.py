from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter
from PIL import Image
from scipy.interpolate import griddata



#team_df = pd.read_csv('2016_team.csv')
#league_df = pd.read_csv('2016_league.csv')
df = pd.read_csv('2016_diff.csv')


def get_diff(team, team_df): 

    team_df = df.loc[team_df['team'] == team] ##'team' decided in the dropdow menu 

    #creating diff: 
    [x,y] = np.round(np.meshgrid(np.linspace(0,85,85),np.linspace(0,100,100)))
    diff = griddata((team_df['y_mid'], team_df['goal_mid']),team_df['raw_diff'],(x,y),method='cubic',fill_value=0)
    #team_rink = griddata((team_df['y_mid'],team_df['goal_mid']),team_df['average_per_hour'],(x,y),method='cubic',fill_value=0)

    #diff = team_rink - league_rink
    #diff = gaussian_filter(diff, sigma=3)

    #team_df['diff'] = df_team.apply(lambda x: get_season_agg(x['y_transformed'], x['goal_dist'], df_league), axis=1)


    x = np.sort(df['y_mid'].unique())
    y = np.sort(df['goal_mid'].unique())
 
    return x, y, diff



app = Dash(__name__) 

app.layout = html.Div([ html.Header("Hockey Shot Maps"), dcc.Dropdown(id="teams_dropdown", options = df["team"].unique(), value = "Anaheim Ducks"), dcc.Graph(id= "shot_map")])



@app.callback(Output("shot_map", "figure"), Input("teams_dropdown", "value"))
def get_map(team): 

    x, y, diff = get_diff(team, df)
    #diff = gaussian_filter(diff, sigma=3)
    print(diff)
    fig = go.Figure(data =
    go.Contour(
        z=diff,
        x=x,
        y=y,
        opacity=0.30,
        colorscale='RdBu'
        
    ))

    fig.update_layout(
        autosize=False,
        width=800,
        height=800)
    fig.update_yaxes(autorange="reversed")
    img = Image.open('semi-nhl-rink.png')
    fig.update_xaxes(range=[0, 85])
    fig.update_yaxes(range=[0, 90])
    
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=x,
            y=-y,
            sizex=86,
            sizey=90,
            sizing="stretch",
            opacity=0.5,
            layer="below")
            )
    

    return fig 



if __name__ == "__main__":
    app.run_server(debug=True)
