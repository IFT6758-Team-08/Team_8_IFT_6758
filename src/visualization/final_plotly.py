
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from PIL import Image
from scipy.interpolate import griddata
import plotly.io as pio
from scipy.ndimage import gaussian_filter

df = pd.read_csv('./2016_difference.csv')

def get_diff(team, team_df): 
    """
    This function takes as input the dataframe of the whole season 
    and returns the corrdinates and the difference in rate at each coordinate for a specific team
    """

    team_df = df.loc[team_df['team'] == team] ##'team' decided in the dropdow menu 

    #creating the diff: 
    x_rink = np.sort(df['y_mid'].unique())
    y_rink = np.sort(df['goal_mid'].unique())

    [x,y] = np.round(np.meshgrid(x_rink,y_rink))
    diff = griddata((team_df['y_mid'], team_df['goal_mid']),team_df['raw_diff'],(x,y),method='cubic',fill_value=0)
 
    return x_rink, y_rink, diff

img = Image.open('semi-nhl-rink.png')


fig = go.Figure()
# Reduce the size of the figure

imae=fig.add_layout_image(
    dict(
            source=img,
            xref="x",
            yref="y",
            x=-43,
            y=-10,
            sizex=85,
            sizey=110,
            #sizing="stretch",
            opacity=1,
            layer="below"
    )
)
#Create a trace for each possible team 
for team in df['team'].unique(): 
    x, y, diff= get_diff(team, df)
    diff = gaussian_filter(diff, sigma=2)
    fig.add_trace(
        go.Contour(
        visible=False,
        z=diff,
        x=x, 
        y=y,
        opacity=0.3,
        colorscale='RdBu',
        #colorscale=[(0, "blue"), (0.5, "white"), (1, "red")],
        name=team
    ))

    

#Resizing the graph: 
fig.update_layout(
    autosize=False,
    width=600,
    height=600, 
    template="simple_white")

# fig.update_yaxes(autorange="reversed")
fig.update_xaxes(range=[-43, 43], title_text='Distance from the center of the rink(ft)')
fig.update_yaxes(range=[90,-10],title_text='Distance from the goal line(ft)')

#Creating a dropdown menu iterating through all the teams:
fig.update_layout(updatemenus=[dict(active=1, 
                  buttons=[dict(args=[{"visible": [i == j for j in range(len(list(df["team"].unique())))]}], 
                                label=team, 
                                method="update") for (i,team) in enumerate(list(df["team"].unique()))],
                                xanchor="right", 
                                yanchor="top")])
# Displaying the title of the 


fig.show()

# Save the figure to html file
#fig.write_html("2019_dropdown.html")
