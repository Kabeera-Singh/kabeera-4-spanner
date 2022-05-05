import random
import pandas as pd
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from networkx.readwrite.adjlist import read_adjlist

import plotly.graph_objects as go
import matplotlib.cm as cm
import copy

def generatorToList(G):
    dct = {}
    for v in G.nodes:
      dct[int(v)] = [int(n) for n in G[v]]
    return dct

def Spanner_4_AlgorithmStep4(G, T ,highDegree , H, dictAdjacency,n):
  """
  Input : 
    T : List of  randomly sampled nodes in T (type int) (has the names of the nodes)
    highDegree : List of high Degree Nodes (list of integers)
    H : Spanner which needs to be updated (networkx graph)
    G : networkx graph
    adjLT : adjacency List  of T , type(adjLT) = Dictionary , key : node name, value : array of neighbors.
    n : number of nodes in graph
  Output: Return P(t,t')


  """
  B = {}
  alreadyPresent = set()
  Limit_High_Degree_Nodes = n**1/5
  for t in T:
    neighborsOfT = dictAdjacency[t]
    if t not in B:
      B[t] = []
    for neighbor in neighborsOfT:
      if neighbor in highDegree and neighbor not in alreadyPresent:
      # if neighbors of node keyT , is a high degree Node
        B[t].append(neighbor)
    # B(T) is now built
  P = {}
  pairs = [] 
  for t in B:
    for t_ in B:
      if t != t_ and (t,t_) not in pairs and (t_,t) not in pairs:
        pairs.append((t,t_))

  for t,t_ in pairs:
    P[(t,t_)] = []
    for u in B[t]:
      for v in B[t_]:
        shortestPath = nx.shortest_path(G, source=u, target=v)
        count_high_deg = len([node for node in shortestPath if node in highDegree])
        if count_high_deg < Limit_High_Degree_Nodes:
          P[(t,t_)].append(shortestPath)
    if len(P[(t,t_)])  == 0:
      continue
    # CHECK , if we want P(t,t_) to be empty or just P to be empty 
    current_length = math.inf
    
    for path in P[(t,t_)]:
      if current_length > len(path):
        shortest_path_p = path
        current_length = len(path)
    nx.add_path(H, shortest_path_p)
  return H

def nx_to_plotly(G,prev_G = None):

    mst = nx.minimum_spanning_tree(G)
    pos = nx.fruchterman_reingold_layout(G, dim=3, seed=42)
    n = G.number_of_edges()
    node_label = list(mst.nodes())

    description = [f"<b>{node} " + "Has " + str(len(list(G.neighbors(node)))) + " edges."
                for node in node_label]
                
    Xnodes = [pos[i][0] for i in G.nodes()]  # x-coordinates of nodes
    Ynodes = [pos[i][1] for i in G.nodes()]  # y-coordinates
    Znodes = [pos[i][2] for i in G.nodes()]  # z-coordinates



    node_colour = []
    for node in node_label:
        if(len(list(G.neighbors(node))) >= n**(2/5)):
            node_colour.append("#e34974")
        else:
            node_colour.append("#71e5eb")

    node_size = 10

    Xedges = []
    Yedges = []
    Zedges = []

    for edge in G.edges():
        #format: [beginning,ending,None]
        x_coords = [pos[edge[0]][0], pos[edge[1]][0], None]
        Xedges += x_coords

        y_coords = [pos[edge[0]][1], pos[edge[1]][1], None]
        Yedges += y_coords

        z_coords = [pos[edge[0]][2], pos[edge[1]][2], None]
        Zedges += z_coords

    # edges
    tracer = go.Scatter3d(x=Xedges, y=Yedges, z=Zedges,
                          mode='lines',
                          line=dict(color="black", width=1),
                          hoverinfo='none',
                          showlegend=False,)

    # nodes
    tracer_marker = go.Scatter3d(x=Xnodes, y=Ynodes, z=Znodes,
                                 mode='markers+text',
                                 textposition='top center',
                                 marker=dict(size=node_size,
                                             line=dict(width=1),
                                             color=node_colour),
                                 hoverinfo='text',
                                 hovertext=description,
                                 # text=node_label,
                                 textfont=dict(size=14, color="black"),
                                 showlegend=False)

    axis_style = dict(title='',
                      titlefont=dict(size=20),
                      showgrid=False,
                      zeroline=False,
                      showline=False,
                      ticks='',
                      showticklabels=False,
                      visible=False)

    layout = go.Layout(

        title={
            'text': "Plot Title",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        width=800,
        height=600,
        showlegend=False,
        scene=dict(xaxis=dict(axis_style),
                   yaxis=dict(axis_style),
                   zaxis=dict(axis_style),
                   ),
        margin=dict(t=100),
        paper_bgcolor='rgb(233,233,233)',
        title_x=0.5,
        hovermode='closest')
    camera = dict(
        up=dict(x=2, y=2, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0, y=0, z=0)
    )

    fig = dict(scene_camera=camera, data=[
               tracer, tracer_marker], layout=layout)

    if prev_G is not None:
            fig = dict(scene_camera=camera, data=
               tracer+ [tracer_marker], layout=layout)
    fig = go.Figure(data=fig["data"],
                    layout=fig["layout"]
                    )
    fig.update_xaxes(showline=True, linewidth=2,
                     linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2,
                     linecolor='black', mirror=True)
    return fig



def scaleNodes(G):
    nodes_lst = G.nodes()
    neighbors = [list(G.neighbors(x)).__len__() for x in nodes_lst]
    BASE_SIZE = 15
    SCALE = 3
    standard_deviation = np.std(list(neighbors))
    mean = np.mean(list(neighbors))
    node_sizes = []
    for node, index in enumerate(nodes_lst):
        node = list(G.neighbors(node)).__len__()
        if mean - 3*standard_deviation < node:
            node_sizes.append(BASE_SIZE+3*SCALE)
        elif mean - 2*standard_deviation < node:
            node_sizes.append(BASE_SIZE+2*SCALE)
        elif mean - standard_deviation < node:
            node_sizes.append(BASE_SIZE+SCALE)
        elif mean >= node:
            node_sizes.append(BASE_SIZE)
        elif mean + standard_deviation < node:
            node_sizes.append(BASE_SIZE-SCALE)
        elif mean + 2*standard_deviation < node:
            node_sizes.append(BASE_SIZE-2*SCALE)
        else:
            node_sizes.append(BASE_SIZE-3*SCALE)
    return node_sizes


def getGraphs(num_nodes, probability):
    G = nx.fast_gnp_random_graph(num_nodes, probability, seed=42)
    # Takes the largest connected component of a graph
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    n = G.number_of_nodes()

    # Initialize H
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    graph_lst = []
    graph_lst.append(nx_to_plotly(G))
    H_lst = []
    H_lst.append(G)

    A = generatorToList(G)

    lowDeg=[]
    highDeg=[]

    def makeLst(adjList):
        compValue=len(adjList)**(2/5)
        #print(compValue)
        for key in adjList:

            edges = [int(x) for x in G.neighbors(int(key))]
        
            if(len(edges)<compValue):
                lowDeg.append(key)
                # add edges for low-degree nodes

            else:
                highDeg.append(key)

    makeLst(A)
    lowDeg = [int(x) for x in lowDeg]
    highDeg = [int(x) for x in highDeg]

    #for node1 in lowDeg:
    #  for node2 in G.neighbors(node1):
    #    H.add_edge(node1, node2) 

    for u in lowDeg:
        for v in A[u]:
            H.add_edge(u,v)

    graph_lst.append(nx_to_plotly(H))
    H_lst.append(copy.deepcopy(H))
    samp = math.ceil( math.pow(n,(2/5))*math.log(n) ) #Random sample n^(2/5)log(n) nodes
    #samp = 3
    S = random.sample(G.nodes, samp)  # sample of O( n^(2/5) * log(n))
    for s in S:  # for each s in the sample construct BFS tree
        bfsEdges = list(nx.bfs_edges(G, s))
        for e in bfsEdges:  # add all edges to H
            # Check if you are adding the same edge twice - ex: (1,2) and (2,1)
            H.add_edge(*e)  # add edges from BFS to H


    graph_lst.append(nx_to_plotly(H))
    H_lst.append(copy.deepcopy(H))
    #print(H.number_of_edges())


    # Randomly samples n^{3/5}*log(n) nodes
    T = random.sample(G.nodes, math.ceil(math.pow(n, (3/5))*math.log(n)))
    # Adds an edge between every high degree node, and the first neighbor in T
    #print("high: ",highDeg.__len__())
    for node in highDeg:
        for neighbor in G.neighbors(node):
            if(neighbor in T):
                H.add_edge(neighbor, node)
                break

    graph_lst.append(nx_to_plotly(H))
    H_lst.append(copy.deepcopy(H))
    #print(H.number_of_edges())

    graph_lst.append(nx_to_plotly(
        Spanner_4_AlgorithmStep4(G, T, highDeg, H, A, n)))
    H_lst.append(Spanner_4_AlgorithmStep4(G, T, highDeg, H, A, n))

    #print("H: ", [x.number_of_edges() for x in H_lst])
    return [graph_lst, H_lst]
