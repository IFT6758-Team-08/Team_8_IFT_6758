import plotly.graph_objects as go
import pandas as pd
import numpy as np

from PIL import Image
from scipy.interpolate import griddata
import plotly.io as pio
from scipy.ndimage import gaussian_filter

df = pd.read_csv('data/2016_diff.csv')

def get_diff(team, team_df): 
    """
    This function takes as input the dataframe of the whole season 
    and returns the corrdinates and the difference in rate at each coordinate for a specific team
    """

    team_df = df.loc[team_df['team'] == team] ##'team' decided in the dropdow menu 

    #creating the diff: 
    x_rink = np.sort(df['y_mid'].unique())
    y_rink = np.sort(df['goal_mid'].unique())
    # Standardize the x between 0 and 80 and the y between 0 and 60
    # x_rink = (x_rink - x_rink.min()) / (x_rink.max() - x_rink.min()) * 90
    # y_rink = (y_rink - y_rink.min()) / (y_rink.max() - y_rink.min()) * 75
    [x,y] = np.round(np.meshgrid(x_rink,y_rink))
    diff = griddata((team_df['y_mid'], team_df['goal_mid']),team_df['raw_diff'],(x,y),method='cubic',fill_value=0)
 
    return x_rink, y_rink, diff

img = Image.open('semi-nhl-rink.png')
# img = img.rotate(90, expand=True)
# img

fig = go.Figure()
# Reduce the size of the figure

imae=fig.add_layout_image(
    dict(
            source=img,
            xref="x",
            yref="y",
            x=00,
            y=-10,
            sizex=85,
            sizey=100,
            # sizing="stretch",
            opacity=1,
            layer="below"
    )
)
#Create a trace for each possible team 
for team in df['team'].unique(): 
    x, y, diff= get_diff(team, df)
    diff = gaussian_filter(diff, sigma=2)
    fig.add_trace(
        # add title for each trace
        go.Contour(
        visible=False,
        z=diff,
        x=x, 
        y=y,
        opacity=0.80,
        colorscale='RdBu',
        name=team
    ))

    

#Resizing the graph: 
fig.update_layout(
    autosize=False,
    width=600,
    height=600)

# fig.update_yaxes(autorange="reversed")
fig.update_xaxes(range=[0, 85], title_text='Distance from the center of the rink(ft)')
fig.update_yaxes(range=[100,-10],title_text='Distance from the goal line(ft)')

#Creating a dropdown menu iterating through all the teams:
fig.update_layout(updatemenus=[dict(active=1, 
                  buttons=[dict(args=[{"visible": [i == j for j in range(len(list(df["team"].unique())))]}], label=team, 
                            method="update") for (i,team) in enumerate(list(df["team"].unique()))])])
# Displaying the title of the 
fig.show()

# Save the figure to html file
fig.write_html("dropdown.html")



""""Kept the file name dynamic so that it can be used for any season"""
file_name = "2016_difference.csv"
df = pd.read_csv('data/'+file_name)

""""Get the year from the file name"""
year = file_name.split("_")[0]

def get_diff(team, team_df): 
    """
    This function takes as input the dataframe of the whole season 
    and returns the corrdinates and the difference in rate at each coordinate for a specific team
    """

    team_df = df.loc[team_df['team'] == team] ##'team' decided in the dropdow menu 

    #creating the diff: 
    x_rink = np.sort(df['y_mid'].unique())
    y_rink = np.sort(df['goal_mid'].unique())

    # [x,y] = np.round(np.meshgrid(x_rink,y_rink))
    [x,y] = np.meshgrid(x_rink,y_rink)
    diff = griddata((team_df['y_mid'], team_df['goal_mid']),team_df['raw_diff'],(x,y),method='cubic',fill_value=0)
 
    return x_rink, y_rink, diff

img = Image.open('references/semi-nhl-rink.png')

"""Create the figure"""
fig = go.Figure()

"""Add the rink image"""
imae=fig.add_layout_image(
    dict(
            source=img,
            xref="x",
            yref="y",
            x=-42,
            y=-10,
            sizex=87,
            sizey=110,
            #sizing = "stretch",
            opacity=1,
            layer="below"
    )
)

"""Create a trace for each possible team"""
for team in df['team'].unique(): 

    x, y, diff= get_diff(team, df)
    diff = gaussian_filter(diff, sigma=2)
    """For color coding"""
    min_diff,max_diff = np.min(diff), np.max(diff)
    if np.abs(min_diff) > np.abs(max_diff):
        max_diff = np.abs(min_diff)
    else:
        min_diff = -np.abs(max_diff)

    fig.add_trace(
        go.Contour(
        visible=False,
        z=diff,
        x=x, 
        y=y,
        opacity=0.7,
        zmin=min_diff,
        zmax=max_diff,
        colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
        name=team
    ))

"""Recise the graph such that the rink and graph are perfectly aligned"""
fig.update_layout(
    autosize=False,
    width=700,
    height=700,
    showlegend=False,
    hovermode='closest'
)

"""Add the x and y axis labels"""
fig.update_xaxes(range=[-42, 41], title_text='Distance from the center of the rink(ft)')
fig.update_yaxes(range=[87,-10],title_text='Distance from the goal line(ft)')

"""Creating a dropdown menu iterating through all the teams"""
fig.update_layout(
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label=team,
                        method="update",
                        args = [{"visible": [i == j for j in range(len(list(df["team"].unique())))]},
                        # Title including the team name and the year
                        {"title": {"text": "Shot Maps for "+team+" for "+year+" - "+str(int(year)+1), "x": 0.5, "xanchor": "center"}}])
                for i, team in enumerate(list(df["team"].unique()))
            ])
        )
    ]
)

"""Make the first team visible with the title"""
fig.update_layout(
    title_text="Shot Maps for "+list(df["team"].unique())[0]+" for "+year+" - "+str(int(year)+1),
    title_x=0.5,
    title_xanchor="center"
)
fig.data[0].visible = True

"""Display the figure"""
fig.show()

"""Save the figure to html file"""
fig.write_html(year+"_dropdown.html")
