
import random
import pandas as pd
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from networkx.readwrite.adjlist import read_adjlist
import plotly.graph_objects as go
import matplotlib.cm as cm
from functions import *


graph_lst, H_lst = [x for x in getGraphs(30,.1)]
graphs_dict = {}
graphs_dict['graph_lst'] = graph_lst
graphs_dict['H_lst'] = H_lst
vars_dict = {}
vars_dict['num_nodes'] = 70
vars_dict['density'] = .5


data = pd.read_csv('data.csv')




# Dash
from jupyter_dash import JupyterDash        
import dash     
import dash_bootstrap_components as dbc
# dash components
from dash import dcc
from dash import html
import dash_daq as daq      
from dash import dash_table
from dash.dependencies import Input, Output, State 
from dash import callback_context
# Data Loading
import joblib
import json
import base64


# Port that the app opens on
port = 8050
graph_counter = 0
with open("styles.json") as json_file:
    styles = json.load(json_file)   # styles.json is a json file containing the styles for the app

HEADER_STYLE = styles["HEADER"]

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = styles["SIDEBAR"]


# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = styles["CONTENT"]
header = html.Div([
    dbc.Row(
            [
            html.H1("Big Data Small Implementation")
            ], justify = "center", align = "center", className = "h-50"
            )
    ], style = HEADER_STYLE
)

sidebar = html.Div( 
    [

        html.H2(
            "Select which graph to view", className = "lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Graph Construction", href = "/",active = "exact"),
                dbc.NavLink("Graph G", href = "/G",active = "exact"),
                dbc.NavLink("Runtime Graphs", href = "/Timing",active = "exact")
            ],
            vertical = True,
            pills = True,
        ),
    ],
    style = SIDEBAR_STYLE,
)
app = JupyterDash(external_stylesheets = [dbc.themes.BOOTSTRAP],suppress_callback_exceptions = True)
server = app.server 
content = html.Div(id = "page-content", style = CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id = "url"),content,header, sidebar])

# Creating the app layouts
Graph_Init_Page = html.Div([
    html.Span("Select the number of nodes and density of the graph"),
    daq.NumericInput(
        id='num-nodes',
        min = 10,
        max = 1000,
        label = "Number of Nodes",
        value = vars_dict['num_nodes']
    ),
    html.Br(),
    daq.NumericInput(
    id='density',
    min = .001,
    max = 1,
    label='Density',
    value = vars_dict['density']
    ),
    html.Div(id='text-out'),
    html.Button("Submit", id = "submit-button",n_clicks=0)
],style = CONTENT_STYLE)

@app.callback(
    Output('text-out', 'children'),
    Input('num-nodes', 'value'),
    Input('density', 'value'),
    Input('submit-button', 'n_clicks')

)

def update_output(nodes,density,button_clicks):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'submit-button' in changed_id:
        vars_dict['num_nodes'] = nodes
        vars_dict['density'] = density
        print("nodes: ", nodes)
        print("density: ", density)
        graphs_dict['graph_lst'], graphs_dict['H_lst'] = [x for x in getGraphs(nodes,density)]
        print("hi2")
        return 'Your graph will have at most {} nodes and have a density of {}.'.format(nodes,density)
    return 'Click submit to generate the graph, then wait for the response to appear, before clicking the graph to view it.'


Graph_Construction_Page = html.Div([
    html.Div([
        html.H2('Spanner',id = "header-text"),
        
    ],style = HEADER_STYLE),
    html.Div([
        html.Span(id = "graph-info"),
    ],style = CONTENT_STYLE),
    html.Br(),
    dcc.Graph(id = "graph_figure",figure = graphs_dict['graph_lst'][0]), # Network correlation Graph
    html.Br(),
    html.Div([
        html.Button("previous",id = "prev",n_clicks=0)],
        style = {'string': 'left', 'display': 'inline-block'}),
    html.Div([
        html.Button("next",id = "next",n_clicks=0)],
        style = {'justify-content':'right', 'display': 'inline-block'})


    ])
@app.callback(
    Output('graph-info','figure'),
        Input('num-nodes', 'value'),
        Input('density', 'value')
)
def update_graph_info(nodes,density):
 return "There are at most {} nodes and is a density of {}".format(vars_dict['num_nodes'],vars_dict['density'])



@app.callback(
    Output('graph_figure','figure'),
    Input('next','n_clicks'),
    Input('prev','n_clicks')
)

def graph_callback(isNext,isPrev):
    graph_counter = abs((isNext - isPrev)%5)
    graph =  graphs_dict['graph_lst'][graph_counter]
    titles = {
        0: "G",
        1: "Step 1",
        2: "Step 2",
        3: "Step 3",
        4: "Step 4 / H",

    }
    graph.update_layout(
        title={
            'text':  titles.get(graph_counter),
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    return graph

Timing_Page = html.Div([
    html.Div([
        html.H2('Timing',id = "header-text"),
        
    ],style = HEADER_STYLE),
    html.Div([
      
        dcc.Graph(id="Timecomplexity", figure =
         go.Figure(data=go.Scatter(x=data["nodes"], y=data["time"],name=str(data["density"]), mode='markers')))
      

    ])
])





@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    pages_dict = {
            "/" : Graph_Init_Page,
            "/G" : Graph_Construction_Page,
            "/Timing" : Timing_Page
            }
    
    if pathname in list(pages_dict.keys()):
        return pages_dict[pathname]


    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className = "text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# Runs application
if __name__ == "__main__": app.run_server(debug=True, port=8050)





