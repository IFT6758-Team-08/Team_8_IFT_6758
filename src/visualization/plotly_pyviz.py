import plotly.graph_objects as go
import pandas as pd
import numpy as np

from PIL import Image
from scipy.interpolate import griddata
import plotly.io as pio




df = pd.read_csv('2016_diff.csv')


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








#io.templates["rink_template"] = go.layout.Template(layout_images=[ dict( name="rink", source="./semi-nhl-rink.png", xref="x", yref="y", x=0, y=0, sizex=100, sizey=100, sizing="stretch", visible=True)])



fig = go.Figure()



#Create a trace for each possible team 
for team in df['team'].unique(): 
    x, y, diff= get_diff(team, df)
    fig.add_trace(
        go.Contour(
        z=diff,
        x=x, 
        y=y,
        opacity=0.30,
        colorscale='RdBu',
        name=team, 
    ))

#Resizing the graph: 
fig.update_layout(
    autosize=False,
    width=800,
    height=800)
fig.update_yaxes(autorange="reversed")
fig.update_xaxes(range=[0, 85])
fig.update_yaxes(range=[-10, 90])
    

#Creating a dropdown menu iterating through all the teams: 
fig.update_layout(updatemenus=[dict(active=0, 
                  buttons=[dict(args=[{"visible": [i == j for j in range(len(list(df["team"].unique())))]}], label=team, 
                            method="update") for (i,team) in enumerate(list(df["team"].unique()))])])

fig.show()
"""
fig.update_layout(
    images=[
        dict(source='./semi-nhl-rink.png', name="Anaheim Ducks", xref="x",
         yref="y",
         sizex=85,
         sizey=90,
         sizing="stretch",
         opacity=1,
         layer="below")
    ]
)







    
app = Dash(__name__) 

app.layout = html.Div([ html.Header("Hockey Shot Maps"), dcc.Dropdown(id="teams_dropdown", options = df["team"].unique(), value = "Anaheim Ducks"), dcc.Graph(id= "shot_map"),  html.H4('Simple plot export options'),
    html.P("↓↓↓ try downloading the plot as PNG ↓↓↓", style={"text-align": "right", "font-weight": "bold"}),
    html.A(
        html.Button("Download as HTML"), 
        id="download",
        href="data:text/html;base64," ,
        download="plotly_graph.html")])



@app.callback(Output("shot_map", "figure"), Input("teams_dropdown", "value"))
def get_map(team): 

    x, y, diff = get_diff(team, df)
    #diff = gaussian_filter(diff, sigma=2)
    fig = px.Figure(data =
    px.Contour(
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
    fig.update_yaxes(range=[-10, 90])
    
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            sizex=86,
            sizey=90,
            sizing="stretch",
            opacity=1,
            layer="below")
            )

    fig.write_html("./file.html")
    return fig 



if __name__ == "__main__":
    main()

"""
