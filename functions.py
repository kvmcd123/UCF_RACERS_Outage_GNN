import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random as rand
import numpy as np
from GATRNN import GATRNN

torch.manual_seed(0)
rand.seed(0)
np.random.seed(0)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''

    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = rand.choice(list(G.nodes))#random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children)!=0:
            dx = width/len(children)
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap,
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# Denormalization Function
def deNormalize(val, mean, range):
    """Denormalizes model output data back to original scale through the following transformation:
        y_denormalized = y_nomalized * y_range + y_mean

        DENORMALIZATION MUST BE DONE BY FEATURE 
        I.E. if normalized output y is (2x100): 
            denormalize y(0,:) and y(1,:) seperately 

    Args:
        val (float): Normalized feature data
        mean (float): Predetermined mean of feature data
        range (float): Predetermined range of feature data

    Returns:
        float: Denormalized feature data
    """
    y = val * range + mean
    return y





# Normalization Function
def normalize(val, mean, range):
    """Normalizes input feature data using the following transformation
    u_normalized = (u_original - u_mean) / u_range

    NORMALIZATION MUST BE DONE BY FEATURE 
    I.E. if original input u is (2x100): 
        denormalize u(0,:) and u(1,:) seperately 

    Args:
        val (float): Original feature data
        mean (float): Predetermined mean of feature data
        range (float): Predetermined range of feature data

    Returns:
        float: Normalized feature data
    """
    y = (val - mean) / range 
    return y





# Function to find the max, mean, and minimum value of feature data passed
def findStats(feature):
    tempMean = []
    tempMax = []
    tempMin = []

    for key in feature:
        tempMean.append(np.mean(key))
        tempMax.append(np.max(key))
        tempMin.append(np.min(key))
    
    meanFeat = np.mean(tempMean)
    maxFeat = np.max(tempMax)
    minFeat = np.min(tempMin)

    return meanFeat,maxFeat,minFeat






# GAT Model Training Function
def trainGAT(model, node_static_features,edge_static_features, node_dynamic_features, edge_index, targets, optimizer, criterion, rangeImpact):
    # Set model into training model
    model.train()
    # Initialize the optimizer's gradient
    optimizer.zero_grad()
    
    # Pass data to model to get estimated output
    outputs = model(node_static_features, edge_static_features, node_dynamic_features, edge_index)
    
    # Calculate the loss of the measured vs estimated output
    loss = criterion(outputs.view(-1, 1)/rangeImpact, targets.view(-1, 1)/rangeImpact)
    
    # Compute the gradient of the loss with respect to all the model parameters
    loss.backward()
    
    # Perform a parameter update
    optimizer.step()
    
    # Return the loss
    return loss.item()






# Model Validation Function
def validGAT(model, node_static_features,edge_static_features, node_dynamic_features, edge_index, targets, optimizer, criterion, rangeImpact):
    # Set the model into evaluation mode
    model.eval()

    # Make sure the gradients are not considered  
    with torch.no_grad():

        # Pass the input data to the model to get estimated output
        outputs = model(node_static_features, edge_static_features, node_dynamic_features, edge_index)

        # Calculate the loss of the measured vs estimated output
        loss = criterion(outputs.view(-1, 1)/rangeImpact, targets.view(-1, 1)/rangeImpact)
    
    # Return the loss
    return loss.item()




# Function to generate wind speed data based on a uniform distribution
def generate_wind_speed_cases(num_cases=10, num_hours=12):
    """
    Generate an array of wind speeds for extreme weather events.

    :param num_cases: Number of extreme weather cases to generate.
    :param num_hours: Number of hours in each case.
    :return: A list of lists, each containing wind speeds for a case.
    """
    extreme_weather_data = []

    for _ in range(num_cases):
        case = [rand.uniform(20, 100) for _ in range(num_hours)]  # wind speeds between 20 mph to 100 mph
        extreme_weather_data.append(case)

    return extreme_weather_data





# Deterministic Impact Level Calculation Function
def generate_impact_level(node, edge, windEvent):
    
    # Set impact to 0 intially
    impact = 0

    # We set higher voltages to have higher impact
    if node.nominalVoltage == 115:
        impact = impact + 3
    elif node.nominalVoltage == 33:
        impact = impact + 2
    elif node.nominalVoltage == 11:
        impact = impact + 1
    else:
        impact = impact

    # If PV then little impact
    if node.busType == 1:
        impact = impact + 1
    # If PQ then most impact
    elif node.busType == 2:
        impact = impact + 2
    # If slack then no impact
    else:
        impact = impact

    # Add the number of loads to impact level
    impact = impact + node.loadsConnected
    
    # If there is an edge considered
    if edge != None:
        
        # If edge is overhead then most impact
        if edge.lineType == 1:
            impact = impact + 2
        #if edge is underground then no impact
        else: 
            impact = impact
        
        # If edge length less than 3 km then little impact
        if edge.lineLength< 3:
            impact = impact + 1
        # if between 3 and 10km then medium impact
        elif edge.lineLength>= 3 and edge.lineLength < 10:
            impact = impact +2
        # if greater than 10 then most impact
        else:
            impact = impact + 3

        # If edge age is new then less impact
        if edge.lineAge < 20:
            impact = impact + 1
        
        # if edge is middle-aged then medium impact,
        elif edge.lineAge >= 20 and edge.lineAge < 30:
            impact = impact + 2
        
        # if edge is old then most impact            
        else:
            impact = impact + 3

    # Find the max wind speed observed in the event
    peakWind = max(windEvent)
    
    # Normalize the wind speed factor by dividing max wind speed by 200
    scaledWind = peakWind/200
    
    # Scale the impact level by normalized wind speed.
    impact = impact*scaledWind

    # Return the impact level
    return impact

