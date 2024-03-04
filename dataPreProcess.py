import torch
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import networkx as nx
import random as rand
from functions import findStats, generate_impact_level, generate_wind_speed_cases
from GraphClasses import Node, Edge
import pickle 

torch.manual_seed(0)
rand.seed(0)

time_steps = 12

# Create an edgelist for topology A and convert to a torch tensor
edgelistA = [(0,1),(0,2),(1,3),(1,4),(4,5),(5,6),(5,7)]
edgelistAt = torch.tensor([[0,0,1,1,4,5,5],  # Source nodes
                           [1,2,3,4,5,6,7]], # Target nodes
                          dtype=torch.long)

# Create the graph for Topology A from edge list
topA = nx.Graph(edgelistA)

# Create custom node objects for Topology A with assigned properties/attributes
An1 = Node("N1", nominalVoltage=115, busType = 0, loadsConnected=0)
An2 = Node("N2", nominalVoltage=33, busType = 1, loadsConnected=1)
An3 = Node("N3", nominalVoltage=33, busType = 1, loadsConnected=3)
An4 = Node("N4", nominalVoltage=11, busType = 1, loadsConnected=3)
An5 = Node("N5", nominalVoltage=11, busType = 2, loadsConnected=3)
An6 = Node("N6", nominalVoltage=11, busType = 2, loadsConnected=10)
An7 = Node("N7", nominalVoltage=0.48, busType = 2, loadsConnected=10)
An8 = Node("N8", nominalVoltage=0.48, busType = 2, loadsConnected=10)

# Create custom edge objects for Topology A with assigned properties/attributes
Ae1 = Edge('E1',lineType= 1, lineLength = 25, lineAge=35)
Ae2 = Edge('E2',lineType= 1, lineLength = 30, lineAge=40)
Ae3 = Edge('E3',lineType= 1, lineLength = 4, lineAge=15)
Ae4 = Edge('E4',lineType= 0, lineLength = 6, lineAge=20)
Ae5 = Edge('E5',lineType= 1, lineLength = 5, lineAge=10)
Ae6 = Edge('E6',lineType= 0, lineLength = 0.7, lineAge=5)
Ae7 = Edge('E7',lineType= 1, lineLength = 0.5, lineAge=3)

# Create a feature array for the nodes in Topology A
A_node_static_features = []
for i in range(1,9):
    A_node_static_features.append([locals()['An'+str(i)].nominalVoltage,locals()['An'+str(i)].busType,locals()['An'+str(i)].loadsConnected])

# Convert the feature array to a torch tensor
A_node_static_features = torch.tensor((A_node_static_features), dtype=torch.float)

# Change the shape of the static features, by simply duplicating them, so that they can be appended with the timeseries data or "dynamic" features
A_node_static_features_repeated = A_node_static_features.unsqueeze(1).repeat(1, time_steps, 1)  # Shape: [nodes, time_steps, 3]

# Create a feature array for the edges in Topology A
A_edge_static_features = []
for i in range(1,8):
    A_edge_static_features.append([locals()['Ae'+str(i)].lineType,locals()['Ae'+str(i)].lineLength,locals()['Ae'+str(i)].lineAge])

# Convert the feature array to a torch tensor
A_edge_static_features = torch.tensor((A_edge_static_features), dtype=torch.float)

# Change the shape of the static features, by simply duplicating them, so that they can be appended with the timeseries data or "dynamic" features
A_edge_static_features_repeated = A_edge_static_features.unsqueeze(1).repeat(1, time_steps, 1)  # Shape: [nodes, time_steps, 3]

# Print the shape of the tensors for confirmation
print(A_node_static_features.shape)
print(A_node_static_features_repeated.shape)
print(A_edge_static_features.shape)
print(A_edge_static_features_repeated.shape)


# Create an edgelist for topology B and convert to a torch tensor
edgelistB = [(0,1),(0,2),(1,3),(1,4),(4,5),(5,6),(5,7)]
edgelistBt = torch.tensor([[0,0,1,1,4,5,5],  # Source nodes
                           [1,2,3,4,5,6,7]], # Target nodes
                          dtype=torch.long)

# Create the graph for Topology B from edge list
topB = nx.Graph(edgelistB)

# Create custom node objects for Topology B with assigned properties/attributes
Bn1 = Node("N1", nominalVoltage=115, busType = 0, loadsConnected=0)
Bn2 = Node("N2", nominalVoltage=33, busType = 1, loadsConnected=0)
Bn3 = Node("N3", nominalVoltage=33, busType = 1, loadsConnected=0)
Bn4 = Node("N4", nominalVoltage=11, busType = 2, loadsConnected=1)
Bn5 = Node("N5", nominalVoltage=33, busType = 1, loadsConnected=4)
Bn6 = Node("N6", nominalVoltage=11, busType = 2, loadsConnected=2)
Bn7 = Node("N7", nominalVoltage=0.48, busType = 2, loadsConnected=20)
Bn8 = Node("N8", nominalVoltage=0.48, busType = 2, loadsConnected=30)

# Create custom edge objects for Topology B with assigned properties/attributes
Be1 = Edge('E1',lineType= 1, lineLength = 30, lineAge=40)
Be2 = Edge('E2',lineType= 1, lineLength = 40, lineAge=20)
Be3 = Edge('E3',lineType= 1, lineLength = 10, lineAge=25)
Be4 = Edge('E4',lineType= 1, lineLength = 20, lineAge=10)
Be5 = Edge('E5',lineType= 0, lineLength = 15, lineAge=5)
Be6 = Edge('E6',lineType= 0, lineLength = 0.5, lineAge=5)
Be7 = Edge('E7',lineType= 0, lineLength = 1, lineAge=5)

# Create a feature array for the nodes in Topology B
B_node_static_features = []
for i in range(1,9):
    B_node_static_features.append([locals()['Bn'+str(i)].nominalVoltage,locals()['Bn'+str(i)].busType,locals()['Bn'+str(i)].loadsConnected])

# Convert the feature array to a torch tensor
B_node_static_features = torch.tensor((B_node_static_features), dtype=torch.float)

# Change the shape of the static features, by simply duplicating them, so that they can be appended with the timeseries data or "dynamic" features
B_node_static_features_repeated = B_node_static_features.unsqueeze(1).repeat(1, time_steps, 1)  # Shape: [nodes, time_steps, 3]

# Create a feature array for the edges in Topology B
B_edge_static_features = []
for i in range(1,8):
    B_edge_static_features.append([locals()['Be'+str(i)].lineType,locals()['Be'+str(i)].lineLength,locals()['Be'+str(i)].lineAge])

# Convert the feature array to a torch tensor
B_edge_static_features = torch.tensor((B_edge_static_features), dtype=torch.float)

# Change the shape of the static features, by simply duplicating them, so that they can be appended with the timeseries data or "dynamic" features
B_edge_static_features_repeated = B_edge_static_features.unsqueeze(1).repeat(1, time_steps, 1)  # Shape: [nodes, time_steps, 3]

# Print the shape of the tensors for confirmation
print(B_node_static_features.shape)
print(B_node_static_features_repeated.shape)
print(B_edge_static_features.shape)
print(B_edge_static_features_repeated.shape)

# Generate the extreme wind data. 
wind_speed_data = generate_wind_speed_cases(num_cases=10,num_hours=time_steps)


# Number of Datasets (10 datasets for Topology A, 10 datasets for Topology B)
nDatasets = 20

# Create empty lists for the training and validation sets
datasets_t = []
datasets_v = []
datasets_vA = []
datasets_vB = []


# Validation Cases for Topology A and Topology B
ValidationCasesA = [3,7]
ValidationCasesB = [3,9]


# DATASET GENERATION LOOP
for i in range(1,11):

    # Convert wind speed data to torch tensor
    locals()["windScenario"+str(i)] =torch.tensor(np.asarray(wind_speed_data[i-1]),dtype = torch.float)
    
    # Create a seperate tensor that accounts for the number of nodes in the topologies
    locals()["windScenario"+str(i)+"N"] =  locals()["windScenario"+str(i)].unsqueeze(0).repeat(8, 1).unsqueeze(2)
    
    # Create a seperate tensor that accounts for the number of edges in the topologies
    locals()["windScenario"+str(i)+"E"] =  locals()["windScenario"+str(i)].unsqueeze(0).repeat(7, 1).unsqueeze(2)

    # Combine the static node and edge features with wind data for Topology A
    locals()["A_node_s"+str(i)+"_features"] = torch.cat((A_node_static_features_repeated, locals()["windScenario"+str(i)+"N"]), dim=2)  # Shape: [nodes, time_steps, 4]
    locals()["A_edge_s"+str(i)+"_features"] = torch.cat((A_edge_static_features_repeated, locals()["windScenario"+str(i)+"E"]), dim=2)  # Shape: [nodes, time_steps, 4]
    
    # Combine the static node and edge features with wind data for Topology B
    locals()["B_node_s"+str(i)+"_features"] = torch.cat((B_node_static_features_repeated, locals()["windScenario"+str(i)+"N"]), dim=2)  # Shape: [nodes, time_steps, 4]
    locals()["B_edge_s"+str(i)+"_features"] = torch.cat((B_edge_static_features_repeated, locals()["windScenario"+str(i)+"E"]), dim=2)  # Shape: [nodes, time_steps, 4]
    
    # Create empty target/output lists
    TargetsA = []
    TargetsB = []

    # Loop through the number of nodes and generate the impact level. 
    # Only immediate parent edges are aggregated into the node level 
    # (i.e. Node 1 --> Edge 1 --> Node 2), 
    # Node 1 Aggregated Impact = Impact (Node 1)
    # Node 2 Aggregated Impact = Impact(Node 2) + Impact (Edge 1)

    for j in range(1,9):
        # If first iteration, only consider the root node
        if j == 1:
            locals()['A_n'+str(j)+'Impact'] = generate_impact_level(locals()['An'+str(j)], None, wind_speed_data[i-1])
            locals()['B_n'+str(j)+'Impact'] = generate_impact_level(locals()['Bn'+str(j)], None, wind_speed_data[i-1])
        # Otherwise consider the current node and its immediate parent edge
        else:
            locals()['A_n'+str(j)+'Impact'] = generate_impact_level(locals()['An'+str(j)], locals()['Ae'+str(j-1)], wind_speed_data[i-1])
            locals()['B_n'+str(j)+'Impact'] = generate_impact_level(locals()['Bn'+str(j)], locals()['Be'+str(j-1)], wind_speed_data[i-1])

        # Append the calculated impact levels for each topology in their respective target lists
        TargetsA.append([locals()['A_n'+str(j)+'Impact']])
        TargetsB.append([locals()['B_n'+str(j)+'Impact']])
    
    # Define the name of the current weather scenario for variable look up (due to locals())
    A_node_scenario = 'A_node_s'+str(i)+'_features'
    A_edge_scenario = 'A_edge_s'+str(i)+'_features'
    
    # Define the name of the current weather scenario for node and edge feature variable look up (due to locals())
    B_node_scenario = 'B_node_s'+str(i)+'_features'
    B_edge_scenario = 'B_edge_s'+str(i)+'_features'
    

    # If the dataset number is in the list of validation cases for Topology A
    if i in ValidationCasesA:
        # Append a dictionary entry containing the validation data to the complete validation set list (contains A and B cases)
        datasets_v.append(
        {
            'topology' :'A',
            'scenario' :i,
            'edge_index':edgelistAt,
            'node_static_features':locals()[A_node_scenario][:, :, :3],
            'node_dynamic_features':locals()[A_node_scenario][:, :, 3:],
            'edge_static_features':locals()[A_edge_scenario][:, :, :3],
            'edge_dynamic_features':locals()[A_edge_scenario][:, :, 3:],
            'targets' :torch.tensor(TargetsA,dtype=torch.float)
        }
        )

        # Append a dictionary entry containing the validation data to the topology A only validation set list
        datasets_vA.append(
        {
            'topology' :'A',
            'scenario' :i,
            'edge_index':edgelistAt,
            'node_static_features':locals()[A_node_scenario][:, :, :3],
            'node_dynamic_features':locals()[A_node_scenario][:, :, 3:],
            'edge_static_features':locals()[A_edge_scenario][:, :, :3],
            'edge_dynamic_features':locals()[A_edge_scenario][:, :, 3:],
            'targets' :torch.tensor(TargetsA,dtype=torch.float)
        }
        )
    # Otherwise append a dictionary entry containing the training data to the training set list 
    else:
        datasets_t.append(
        {
            'topology' :'A',
            'scenario' :i,
            'edge_index':edgelistAt,
            'node_static_features':locals()[A_node_scenario][:, :, :3],
            'node_dynamic_features':locals()[A_node_scenario][:, :, 3:],
            'edge_static_features':locals()[A_edge_scenario][:, :, :3],
            'edge_dynamic_features':locals()[A_edge_scenario][:, :, 3:],
            'targets' :torch.tensor(TargetsA,dtype=torch.float)
        }
        )

    # If the dataset number is in the list of validation cases for Topology B
    if i in ValidationCasesB:
        # Append a dictionary entry containing the validation data to the complete validation set list (contains A and B cases)
        datasets_v.append(
        {
            'topology' :'B',
            'scenario' :i,
            'edge_index':edgelistBt,
            'node_static_features':locals()[B_node_scenario][:, :, :3],
            'node_dynamic_features':locals()[B_node_scenario][:, :, 3:],
            'edge_static_features':locals()[B_edge_scenario][:, :, :3],
            'edge_dynamic_features':locals()[B_edge_scenario][:, :, 3:],
            'targets' :torch.tensor(TargetsB,dtype=torch.float)
        }
        )

        # Append a dictionary entry containing the validation data to the topology A only validation set list
        datasets_vB.append(
        {
            'topology' :'B',
            'scenario' :i,
            'edge_index':edgelistBt,
            'node_static_features':locals()[B_node_scenario][:, :, :3],
            'node_dynamic_features':locals()[B_node_scenario][:, :, 3:],
            'edge_static_features':locals()[B_edge_scenario][:, :, :3],
            'edge_dynamic_features':locals()[B_edge_scenario][:, :, 3:],
            'targets' :torch.tensor(TargetsB,dtype=torch.float)
        }
        )

    # Otherwise append a dictionary entry containing the training data to the training set list 
    else:
        datasets_t.append(
        {
            'topology' :'B',
            'scenario' :i,
            'edge_index':edgelistBt,
            'node_static_features':locals()[B_node_scenario][:, :, :3],
            'node_dynamic_features':locals()[B_node_scenario][:, :, 3:],
            'edge_static_features':locals()[B_edge_scenario][:, :, :3],
            'edge_dynamic_features':locals()[B_edge_scenario][:, :, 3:],
            'targets' :torch.tensor(TargetsB,dtype=torch.float)
        }
        )
        
# Create an empty list to put all the datasets in.
# This will be used to find the data normalization statistics
datasets = []

# Append the training datasets into the complete dataset list
for key in datasets_t:
  datasets.append(key)

# Append the validation datasets into the complete dataset list
for key in datasets_v:
  datasets.append(key)

# Print lengths of all datasets for confirmation
print(len(datasets_t))
print(len(datasets_v))
print(len(datasets_vA))
print(len(datasets_vB))
print(len(datasets))

# Create empty lists to store data for each feature
busVoltage = []
busType = []
busLoads = []
lineAge = []
lineType = []
lineSpan = []
windSpeed = []
impactLevel = []

# Loop through each dataset and append the node and edge features, weather features, and targets to their respective lists
# The tensors are converted to numpy arrays beforehand 
for key in datasets:
    busVoltage.append(key['node_static_features'][:,0,0].numpy())
    busType.append(key['node_static_features'][:,0,1].numpy())
    busLoads.append(key['node_static_features'][:,0,2].numpy())
    
    lineType.append(key['edge_static_features'][:,0,0].numpy())
    lineSpan.append(key['edge_static_features'][:,0,1].numpy())
    lineAge.append(key['edge_static_features'][:,0,2].numpy())

    windSpeed.append(key['node_dynamic_features'][0,:,0].numpy())
    impactLevel.append(key['targets'].numpy())


# Determine the mean, and the range (max-min) of each node feature over all the data
meanBVolt, maxBVolt, minBVolt = findStats(busVoltage)
meanBType, maxBType, minBType = findStats(busType)
meanBLoads, maxBLoads, minBLoads = findStats(busLoads)

# Determine the mean, and the range (max-min) of each edge feature over all the data
meanLType, maxLType, minLType = findStats(lineType)
meanLSpan, maxLSpan, minLSpan = findStats(lineSpan)
meanLAge, maxLAge, minLAge = findStats(lineAge)

# Determine the mean, and the range (max-min) of each weather feature over all the data
meanSpeed, maxSpeed, minSpeed = findStats(windSpeed)

# Determine the mean, and the range (max-min) of each target over all the data
meanImpact, maxImpact, minImpact = findStats(impactLevel)

# Create an array of names to assign normalization statistics to
variables = ['BVolt','BType','BLoads','LType','LSpan','LAge','Speed','Impact']

# Create an empty dictionary to store the normalization statistics in
results = {}

# Loop through each variable
for key in variables:
    # Set the mean of the current feature
    results["mean" + key] =locals()['mean'+key]
    # Set the range of the current feature
    results["range" + key] = locals()['max'+key] - locals()['min'+key]
    
    # Use the following check to take into account if there is only one unique value in all the data for a certain variable. 
    # Since Python starts from 0, one unique value corresponds to 0 and must be changed to 1 so division errors are not encountered
    # if results["range" + key] == 0:
    #     print(results["mean" + key])
    #     print(results["range" + key])
    #     print(locals()['mean'+key])

# Set the file name that the scaling/normalization factors will be saved to
filename = os.path.join("sF_dict.csv")

# Open the file in 'w' mode to create an empty file
with open(filename, 'w', newline='') as csvfile:
    
    # Create a CSV writer object
    w = csv.writer(csvfile)
    
    # Save the scaling factors
    for key, val in results.items():
        w.writerow([key, val])

# Create a dictionary to save all the datasets in, along with scaling factors
datasetDict = {
    'train':datasets_t,
    'validate':datasets_v,
    'validateA':datasets_vA,
    'validateB':datasets_vB,
    'sF': results
}

# Save the dictionary in a pickle file to be loaded later
with open('dataDict.pkl', 'wb') as f:
    pickle.dump(datasetDict, f)
  



    
    