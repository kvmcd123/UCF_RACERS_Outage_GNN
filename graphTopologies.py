
import torch
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import networkx as nx
import random as rand
from functions import hierarchy_pos,findStats, generate_impact_level, generate_wind_speed_cases
from GraphClasses import Node, Edge
import pickle 
import matplotlib.pyplot as plt
import networkx as nx
from itertools import chain

# Function to determine the line type string based on its type value
def get_line_type_string(line_type):
    return 'Underground' if line_type == 0 else 'Overhead'


# Function to determine the bus type string based on its type value
def get_bus_type_string(bus_type):
    if bus_type == 0:
        return 'Slack'
    elif bus_type == 1:
        return 'PV'
    elif bus_type == 2:
        return 'PQ'
    else:
        return 'Unknown'


# Function to extract labels for plotting
def extract_labels(topA,attribute):
    return {i: topA.nodes[i][attribute] for i in topA.nodes()}

torch.manual_seed(0)
rand.seed(0)
np.random.seed(0)

# we create a topology, topA from an edgelist, edgelistA
edgelistA = [(0,1),(0,2),(1,3),(1,4),(4,5),(5,6),(5,7)]
edgelistAt = torch.tensor([[0,0,1,1,4,5,5],  # Source nodes
                           [1,2,3,4,5,6,7]], # Target nodes
                          dtype=torch.long)

n1 = Node("N1", nominalVoltage=115, busType = 0, loadsConnected=0)
n2 = Node("N2", nominalVoltage=33, busType = 1, loadsConnected=1)
n3 = Node("N3", nominalVoltage=33, busType = 1, loadsConnected=3)
n4 = Node("N4", nominalVoltage=11, busType = 1, loadsConnected=3)
n5 = Node("N5", nominalVoltage=11, busType = 2, loadsConnected=3)
n6 = Node("N6", nominalVoltage=11, busType = 2, loadsConnected=10)
n7 = Node("N7", nominalVoltage=0.48, busType = 2, loadsConnected=10)
n8 = Node("N8", nominalVoltage=0.48, busType = 2, loadsConnected=10)

e1 = Edge('E1',lineType= 1, lineLength = 25, lineAge=35)
e2 = Edge('E2',lineType= 1, lineLength = 30, lineAge=40)
e3 = Edge('E3',lineType= 1, lineLength = 4, lineAge=15)
e4 = Edge('E4',lineType= 0, lineLength = 6, lineAge=20)
e5 = Edge('E5',lineType= 1, lineLength = 5, lineAge=10)
e6 = Edge('E6',lineType= 0, lineLength = 0.7, lineAge=5)
e7 = Edge('E7',lineType= 1, lineLength = 0.5, lineAge=3)


# Create the nodes and edges
nodes = [n1, n2, n3,n4,n5,n6,n7,n8]
edges = [e1, e2, e3,e4,e5,e6,e7]

# Create the networkx graph from the edgelist
topA = nx.Graph()
topA.add_edges_from(edgelistA)

# Assign attributes to nodes
for i, node in enumerate(nodes):
    topA.nodes[i]["name"] = node.name
    topA.nodes[i]["nominalVoltage"] = node.nominalVoltage
    topA.nodes[i]["busType"] = node.busType
    topA.nodes[i]["loadsConnected"] = node.loadsConnected

# Assign attributes to edges
for i, (src, dest) in enumerate(edgelistA):
    topA[src][dest]["name"] = edges[i].name
    topA[src][dest]["lineType"] = edges[i].lineType
    topA[src][dest]["lineLength"] = edges[i].lineLength
    topA[src][dest]["lineAge"] = edges[i].lineAge


# Plot the network
pos = hierarchy_pos(topA, 0)

node_labels = {node: f"{data['name']}\n"
                        f"Voltage: {data['nominalVoltage']}kV\n"
                        
                        f"Type: {data['busType']}\n"
                        f"Loads: {data['loadsConnected']}"
               for node, data in topA.nodes(data=True)}



# Create a label for each node's additional attributes to display at the side
side_labels = {node: f"Volt: {data['nominalVoltage']}kV\n"
                       f"Type: {get_bus_type_string(data['busType'])}\n"
                       f"Loads: {data['loadsConnected']}"
              for node, data in topA.nodes(data=True)}

# Create a dictionary for node names to display them at the center of the node
node_names = {node: data['name'] for node, data in topA.nodes(data=True)}


# Create a label for each edge to display near the edge
edge_attributes = {(src, dest): f"{data['name']}\n"
                                f"Type: {get_line_type_string(data['lineType'])}\n"
                                f"Length: {data['lineLength']}km\n"
                                f"Age: {data['lineAge']} years"
                   for src, dest, data in topA.edges(data=True)}

# Create a label for each edge that contains its name only
edge_names = {(src, dest): data['name'] for src, dest, data in topA.edges(data=True)}

# Create a separate label for the other edge attributes to display near the edge but not on top
edge_side_labels = {(src, dest): f"Type: {get_line_type_string(data['lineType'])}\n"
                                 f"Length: {data['lineLength']}km\n"
                                 f"Age: {data['lineAge']} years"
                    for src, dest, data in topA.edges(data=True)}

# Draw the network
plt.figure(constrained_layout=True)#figsize=(12, 8))
plt.title('Topology A')

# Draw nodes and node labels
nx.draw(topA, pos, with_labels=False, node_color='lightblue', edge_color='gray')
nx.draw_networkx_labels(topA, pos, labels=node_names)

# Draw edges# Draw the side labels with the node attributes
for node, label in side_labels.items():
    x, y = pos[node]
    plt.text(x + 0.015, y, label, fontsize=8, verticalalignment='center', horizontalalignment='left')

nx.draw_networkx_edges(topA, pos, edge_color='gray')

# Draw edge names at the center of the edges
nx.draw_networkx_edge_labels(topA, pos, edge_labels=edge_names)

# Draw the edge side labels with the other edge attributes
for (src, dest), label in edge_side_labels.items():
    # Get the positions of the source and destination nodes
    x0, y0 = pos[src]
    x1, y1 = pos[dest]
    # Calculate the midpoint for the edge name
    mid_x, mid_y = ((x0 + x1) / 2, (y0 + y1) / 2)
    # Offset the position for the additional attributes
    # This offset is somewhat arbitrary and may need adjusting
    offset_x, offset_y = (x1 - x0) * 0.1, (y1 - y0) * 0.1
    label_pos = (mid_x + offset_x, mid_y + offset_y)
    # plt.text(label_pos[0], label_pos[1], label, fontsize=8,
    #          verticalalignment='center', horizontalalignment='center')
    plt.text(mid_x+0.02, mid_y, label, fontsize=8,
             verticalalignment='center', horizontalalignment='left')

plt.show()


# we create a topology, topB from an edgelist, edgelistB
edgelistB = [(0,1),(0,2),(1,3),(1,4),(2,5),(3,6),(5,7)]
edgelistBt = torch.tensor([[0,0,1,1,2,3,5],  # Source nodes
                           [1,2,3,4,5,6,7]], # Target nodes
                          dtype=torch.long)
topB = nx.Graph(edgelistB)
pos = hierarchy_pos(topB,0)

Bn1 = Node("N1", nominalVoltage=115, busType = 0, loadsConnected=0)
Bn2 = Node("N2", nominalVoltage=33, busType = 1, loadsConnected=0)
Bn3 = Node("N3", nominalVoltage=33, busType = 1, loadsConnected=0)
Bn4 = Node("N4", nominalVoltage=11, busType = 2, loadsConnected=1)
Bn5 = Node("N5", nominalVoltage=33, busType = 1, loadsConnected=4)
Bn6 = Node("N6", nominalVoltage=11, busType = 2, loadsConnected=2)
Bn7 = Node("N7", nominalVoltage=0.48, busType = 2, loadsConnected=20)
Bn8 = Node("N8", nominalVoltage=0.48, busType = 2, loadsConnected=30)

Be1 = Edge('E1',lineType= 1, lineLength = 30, lineAge=40)
Be2 = Edge('E2',lineType= 1, lineLength = 40, lineAge=20)
Be3 = Edge('E3',lineType= 1, lineLength = 10, lineAge=25)
Be4 = Edge('E4',lineType= 1, lineLength = 20, lineAge=10)
Be5 = Edge('E5',lineType= 0, lineLength = 15, lineAge=5)
Be6 = Edge('E6',lineType= 0, lineLength = 0.5, lineAge=5)
Be7 = Edge('E7',lineType= 0, lineLength = 1, lineAge=5)

B_node_static_features = []
for i in range(1,9):
    B_node_static_features.append([locals()['Bn'+str(i)].nominalVoltage,locals()['Bn'+str(i)].busType,locals()['Bn'+str(i)].loadsConnected])


B_node_static_features = torch.tensor((B_node_static_features), dtype=torch.float)

time_steps = 12
B_node_static_features_repeated = B_node_static_features.unsqueeze(1).repeat(1, time_steps, 1)  # Shape: [nodes, time_steps, 3]


B_edge_static_features = []
for i in range(1,8):
    B_edge_static_features.append([locals()['Be'+str(i)].lineType,locals()['Be'+str(i)].lineLength,locals()['Be'+str(i)].lineAge])


B_edge_static_features = torch.tensor((B_edge_static_features), dtype=torch.float)

time_steps = 12
B_edge_static_features_repeated = B_edge_static_features.unsqueeze(1).repeat(1, time_steps, 1)  # Shape: [nodes, time_steps, 3]

# Create the nodes and edges
nodes = [Bn1, Bn2, Bn3,Bn4,Bn5,Bn6,Bn7,Bn8]
edges = [Be1, e2, Be3,Be4,Be5,Be6,Be7]

# Create the networkx graph from the edgelist
topB = nx.Graph()
topB.add_edges_from(edgelistB)

# Assign attributes to nodes
for i, node in enumerate(nodes):
    topB.nodes[i]["name"] = node.name
    topB.nodes[i]["nominalVoltage"] = node.nominalVoltage
    topB.nodes[i]["busType"] = node.busType
    topB.nodes[i]["loadsConnected"] = node.loadsConnected

# Assign attributes to edges
for i, (src, dest) in enumerate(edgelistB):
    topB[src][dest]["name"] = edges[i].name
    topB[src][dest]["lineType"] = edges[i].lineType
    topB[src][dest]["lineLength"] = edges[i].lineLength
    topB[src][dest]["lineAge"] = edges[i].lineAge

node_labels = {node: f"{data['name']}\n"
                        f"Voltage: {data['nominalVoltage']}kV\n"
                        
                        f"Type: {data['busType']}\n"
                        f"Loads: {data['loadsConnected']}"
               for node, data in topB.nodes(data=True)}

# Create a label for each node's additional attributes to display at the side
side_labels = {node: f"Volt: {data['nominalVoltage']}kV\n"
                       f"Type: {get_bus_type_string(data['busType'])}\n"
                       f"Loads: {data['loadsConnected']}"
              for node, data in topA.nodes(data=True)}

# Create a dictionary for node names to display them at the center of the node
node_names = {node: data['name'] for node, data in topA.nodes(data=True)}


# Create a label for each edge to display near the edge
edge_attributes = {(src, dest): f"{data['name']}\n"
                                f"Type: {get_line_type_string(data['lineType'])}\n"
                                f"Length: {data['lineLength']}km\n"
                                f"Age: {data['lineAge']} years"
                   for src, dest, data in topA.edges(data=True)}


# Create a label for each edge that contains its name only
edge_names = {(src, dest): data['name'] for src, dest, data in topB.edges(data=True)}

# Create a separate label for the other edge attributes to display near the edge but not on top
edge_side_labels = {(src, dest): f"Type: {get_line_type_string(data['lineType'])}\n"
                                 f"Length: {data['lineLength']}km\n"
                                 f"Age: {data['lineAge']} years"
                    for src, dest, data in topA.edges(data=True)}

# Draw the network
plt.figure(constrained_layout=True)#figsize=(12, 8))
plt.title('Topology B')

# Draw nodes and node labels
nx.draw(topB, pos, with_labels=False, node_color='lightblue', edge_color='gray')
nx.draw_networkx_labels(topB, pos, labels=node_names)

# Draw edges# Draw the side labels with the node attributes
for node, label in side_labels.items():
    x, y = pos[node]
    plt.text(x + 0.015, y, label, fontsize=8, verticalalignment='center', horizontalalignment='left')

nx.draw_networkx_edges(topB, pos, edge_color='gray')

# Draw edge names at the center of the edges
nx.draw_networkx_edge_labels(topB, pos, edge_labels=edge_names)

# Draw the edge side labels with the other edge attributes
for (src, dest), label in edge_side_labels.items():
    # Get the positions of the source and destination nodes
    x0, y0 = pos[src]
    x1, y1 = pos[dest]
    # Calculate the midpoint for the edge name
    mid_x, mid_y = ((x0 + x1) / 2, (y0 + y1) / 2)
    # Offset the position for the additional attributes
    # This offset is somewhat arbitrary and may need adjusting
    offset_x, offset_y = (x1 - x0) * 0.1, (y1 - y0) * 0.1
    label_pos = (mid_x + offset_x, mid_y + offset_y)
    # plt.text(label_pos[0], label_pos[1], label, fontsize=8,
    #          verticalalignment='center', horizontalalignment='center')
    plt.text(mid_x+0.02, mid_y, label, fontsize=8,
             verticalalignment='center', horizontalalignment='left')

plt.show()



# Example usage
plt.figure()
wind_speed_data = generate_wind_speed_cases()
for i, case in enumerate(wind_speed_data):
    plt.plot(np.linspace(0,12,12),case)
xlabels = ['1:00PM','2:00PM','3:00PM','4:00PM','5:00PM','6:00PM','7:00PM','8:00PM','9:00PM','10:00PM','11:00PM','12:00AM']
plt.xticks(np.linspace(0,12,12),xlabels)
plt.xlabel('Time')
plt.ylabel('Wind Speed (mph)')
plt.title('Extreme Weather Events')
plt.show()