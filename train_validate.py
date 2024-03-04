import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import random as rand
import numpy as np
from functions import trainGAT, validGAT, normalize,deNormalize
from GATRNN import GATRNN
import pickle 

torch.manual_seed(0)
rand.seed(0)
np.random.seed(0)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


#########################################################################################################################
#                                           DATA-PRE-PROCESS PORTION
#########################################################################################################################
# Load the data from the saved pickle file
with open('dataDict.pkl', 'rb') as f:
    datasetsDict = pickle.load(f)

# Assign each dataset to it's respective variable
datasets_t = datasetsDict['train']
datasets_v = datasetsDict['validate']
datasets_vA = datasetsDict['validateA']
datasets_vB = datasetsDict['validateB']
sF = datasetsDict['sF']

# Create a new dataset list to store the normalized data in
ndatasets_t = []

# Loop through training data
for key in datasets_t:
    # Aquire numpy arrays of each tensor for node, edge, weather and target features
    busVolt = np.copy(key['node_static_features'][:,:,0].detach().numpy())
    busType = np.copy(key['node_static_features'][:,:,1].detach().numpy())
    busLoads  = np.copy(key['node_static_features'][:,:,2].detach().numpy())
    
    lineType = np.copy(key['edge_static_features'][:,:,0].detach().numpy())
    lineSpan = np.copy(key['edge_static_features'][:,:,1].detach().numpy())
    lineAge  = np.copy(key['edge_static_features'][:,:,2].detach().numpy())
    windSpeed = np.copy(key['node_dynamic_features'][:,:,:].detach().numpy())
    impactLevel = np.copy(key['targets'][:,0].detach().numpy())

    # Normalize the node feature data using the mean and range
    nbusVolt = normalize(busVolt,float(sF['meanBVolt']),float(sF['rangeBVolt']))
    nbusType = normalize(busType,float(sF['meanBType']),float(sF['rangeBType']))
    nbusLoad = normalize(busLoads,float(sF['meanBLoads']),float(sF['rangeBLoads']))

    # Normalize the edge feature data using the mean and range
    nlineType = normalize(lineType,float(sF['meanLType']),float(sF['rangeLType']))
    nlineSpan = normalize(lineSpan,float(sF['meanLSpan']),float(sF['rangeLSpan']))
    nlineAge = normalize(lineAge,float(sF['meanLAge']),float(sF['rangeLAge']))
    
    # Normalize the weather feature data using the mean and range
    nwindSpeed = normalize(windSpeed,float(sF['meanSpeed']),float(sF['rangeSpeed']))
    
    # Normalize the target feature data using the mean and range
    nimpactLevel = normalize(impactLevel,float(sF['meanImpact']),float(sF['rangeImpact']))

    # Combine all normalized node features into array
    staticNP = np.asarray([nbusVolt, nbusType, nbusLoad]).transpose((1, 2, 0))
    
    # Combine all normalized edge features into array
    staticEP = np.asarray([nlineType, nlineSpan, nlineAge]).transpose((1, 2, 0))
    
    # Convert normalized node array to tensor
    staticN =torch.tensor(staticNP, dtype=torch.float)
    
    # Convert normalized edge array to tensor
    staticE =torch.tensor(staticEP, dtype=torch.float)
    
    # Convert wind feature array to tensor
    dynamic =torch.tensor(nwindSpeed, dtype=torch.float)
    
    # Convert target array to tensor
    targets =torch.tensor(nimpactLevel, dtype=torch.float).t()
    
    # Append all the data into the dictionary entry similar to before
    ndatasets_t.append(
          {
            'topology' :key['topology'],
            'scenario' :key['scenario'],
            'edge_index':key['edge_index'],
            'node_static_features': staticN,
            'edge_static_features': staticE,
            'node_dynamic_features':dynamic,
            'targets' :targets
          }
        )
    
# Loop through complete validation data
ndatasets_v = []
for key in datasets_v:
    # Aquire numpy arrays of each tensor for node, edge, weather and target features
    busVolt = np.copy(key['node_static_features'][:,:,0].detach().numpy())
    busType = np.copy(key['node_static_features'][:,:,1].detach().numpy())
    busLoads  = np.copy(key['node_static_features'][:,:,2].detach().numpy())
    
    lineType = np.copy(key['edge_static_features'][:,:,0].detach().numpy())
    lineSpan = np.copy(key['edge_static_features'][:,:,1].detach().numpy())
    lineAge  = np.copy(key['edge_static_features'][:,:,2].detach().numpy())
    windSpeed = np.copy(key['node_dynamic_features'][:,:,:].detach().numpy())
    impactLevel = np.copy(key['targets'][:,0].detach().numpy())

    # Normalize the node feature data using the mean and range
    nbusVolt = normalize(busVolt,float(sF['meanBVolt']),float(sF['rangeBVolt']))
    nbusType = normalize(busType,float(sF['meanBType']),float(sF['rangeBType']))
    nbusLoad = normalize(busLoads,float(sF['meanBLoads']),float(sF['rangeBLoads']))

    # Normalize the edge feature data using the mean and range
    nlineType = normalize(lineType,float(sF['meanLType']),float(sF['rangeLType']))
    nlineSpan = normalize(lineSpan,float(sF['meanLSpan']),float(sF['rangeLSpan']))
    nlineAge = normalize(lineAge,float(sF['meanLAge']),float(sF['rangeLAge']))
    
    # Normalize the weather feature data using the mean and range
    nwindSpeed = normalize(windSpeed,float(sF['meanSpeed']),float(sF['rangeSpeed']))
    
    # Normalize the target feature data using the mean and range
    nimpactLevel = normalize(impactLevel,float(sF['meanImpact']),float(sF['rangeImpact']))
    
    # Combine all normalized node features into array
    staticNP = np.asarray([nbusVolt, nbusType, nbusLoad]).transpose((1, 2, 0))
    staticEP = np.asarray([nlineType, nlineSpan, nlineAge]).transpose((1, 2, 0))
    
    # Convert normalized node array to tensor
    staticN =torch.tensor(staticNP, dtype=torch.float)
    
    # Convert normalized edge array to tensor
    staticE =torch.tensor(staticEP, dtype=torch.float)
    
    # Convert wind feature array to tensor
    dynamic =torch.tensor(nwindSpeed, dtype=torch.float)
    
    # Convert target array to tensor
    targets =torch.tensor(nimpactLevel, dtype=torch.float).t()
        
    # Append all the data into the dictionary entry similar to before
    ndatasets_v.append(
          {
            'topology' :key['topology'],
            'scenario' :key['scenario'],
            'edge_index':key['edge_index'],
            'node_static_features': staticN,
            'edge_static_features': staticE,
            'node_dynamic_features':dynamic,
            'targets' :targets
          }
        )

# Create empty list for validation A data
ndatasets_vA = []

# Loop through validation A data
for key in datasets_vA:    
    # Aquire numpy arrays of each tensor for node, edge, weather and target features
    busVolt = np.copy(key['node_static_features'][:,:,0].detach().numpy())
    busType = np.copy(key['node_static_features'][:,:,1].detach().numpy())
    busLoads  = np.copy(key['node_static_features'][:,:,2].detach().numpy())
    
    lineType = np.copy(key['edge_static_features'][:,:,0].detach().numpy())
    lineSpan = np.copy(key['edge_static_features'][:,:,1].detach().numpy())
    lineAge  = np.copy(key['edge_static_features'][:,:,2].detach().numpy())
    windSpeed = np.copy(key['node_dynamic_features'][:,:,:].detach().numpy())
    impactLevel = np.copy(key['targets'][:,0].detach().numpy())

    # Normalize the node feature data using the mean and range
    nbusVolt = normalize(busVolt,float(sF['meanBVolt']),float(sF['rangeBVolt']))
    nbusType = normalize(busType,float(sF['meanBType']),float(sF['rangeBType']))
    nbusLoad = normalize(busLoads,float(sF['meanBLoads']),float(sF['rangeBLoads']))

    # Normalize the edge feature data using the mean and range
    nlineType = normalize(lineType,float(sF['meanLType']),float(sF['rangeLType']))
    nlineSpan = normalize(lineSpan,float(sF['meanLSpan']),float(sF['rangeLSpan']))
    nlineAge = normalize(lineAge,float(sF['meanLAge']),float(sF['rangeLAge']))
    
    # Normalize the weather feature data using the mean and range
    nwindSpeed = normalize(windSpeed,float(sF['meanSpeed']),float(sF['rangeSpeed']))
    
    # Normalize the target feature data using the mean and range
    nimpactLevel = normalize(impactLevel,float(sF['meanImpact']),float(sF['rangeImpact']))
    
    # Combine all normalized node features into array
    staticNP = np.asarray([nbusVolt, nbusType, nbusLoad]).transpose((1, 2, 0))
    staticEP = np.asarray([nlineType, nlineSpan, nlineAge]).transpose((1, 2, 0))
    
    # Convert normalized node array to tensor
    staticN =torch.tensor(staticNP, dtype=torch.float)
    
    # Convert normalized edge array to tensor
    staticE =torch.tensor(staticEP, dtype=torch.float)
    
    # Convert wind feature array to tensor
    dynamic =torch.tensor(nwindSpeed, dtype=torch.float)
    
    # Convert target array to tensor
    targets =torch.tensor(nimpactLevel, dtype=torch.float).t()

    # Append all the data into the dictionary entry similar to before
    ndatasets_vA.append(
          {
            'topology' :key['topology'],
            'scenario' :key['scenario'],
            'edge_index':key['edge_index'],
            'node_static_features': staticN,
            'edge_static_features': staticE,
            'node_dynamic_features':dynamic,
            'targets' :targets
          }
        )

# Create empty list for validation B data
ndatasets_vB = []

# Loop through validation B data
for key in datasets_vB:
    # Aquire numpy arrays of each tensor for node, edge, weather and target features
    busVolt = np.copy(key['node_static_features'][:,:,0].detach().numpy())
    busType = np.copy(key['node_static_features'][:,:,1].detach().numpy())
    busLoads  = np.copy(key['node_static_features'][:,:,2].detach().numpy())
    
    lineType = np.copy(key['edge_static_features'][:,:,0].detach().numpy())
    lineSpan = np.copy(key['edge_static_features'][:,:,1].detach().numpy())
    lineAge  = np.copy(key['edge_static_features'][:,:,2].detach().numpy())
    windSpeed = np.copy(key['node_dynamic_features'][:,:,:].detach().numpy())
    impactLevel = np.copy(key['targets'][:,0].detach().numpy())

    # Normalize the node feature data using the mean and range
    nbusVolt = normalize(busVolt,float(sF['meanBVolt']),float(sF['rangeBVolt']))
    nbusType = normalize(busType,float(sF['meanBType']),float(sF['rangeBType']))
    nbusLoad = normalize(busLoads,float(sF['meanBLoads']),float(sF['rangeBLoads']))

    # Normalize the edge feature data using the mean and range
    nlineType = normalize(lineType,float(sF['meanLType']),float(sF['rangeLType']))
    nlineSpan = normalize(lineSpan,float(sF['meanLSpan']),float(sF['rangeLSpan']))
    nlineAge = normalize(lineAge,float(sF['meanLAge']),float(sF['rangeLAge']))
    
    # Normalize the weather feature data using the mean and range
    nwindSpeed = normalize(windSpeed,float(sF['meanSpeed']),float(sF['rangeSpeed']))
    
    # Normalize the target feature data using the mean and range
    nimpactLevel = normalize(impactLevel,float(sF['meanImpact']),float(sF['rangeImpact']))
    
    # Combine all normalized node features into array
    staticNP = np.asarray([nbusVolt, nbusType, nbusLoad]).transpose((1, 2, 0))
    staticEP = np.asarray([nlineType, nlineSpan, nlineAge]).transpose((1, 2, 0))
    
    # Convert normalized node array to tensor
    staticN =torch.tensor(staticNP, dtype=torch.float)
    
    # Convert normalized edge array to tensor
    staticE =torch.tensor(staticEP, dtype=torch.float)
    
    # Convert wind feature array to tensor
    dynamic =torch.tensor(nwindSpeed, dtype=torch.float)
    
    # Convert target array to tensor
    targets =torch.tensor(nimpactLevel, dtype=torch.float).t()
    
    # Append all the data into the dictionary entry similar to before
    ndatasets_vB.append(
          {
            'topology' :key['topology'],
            'scenario' :key['scenario'],
            'edge_index':key['edge_index'],
            'node_static_features': staticN,
            'edge_static_features': staticE,
            'node_dynamic_features':dynamic,
            'targets' :targets
          }
        )

# Print length of training dataset for verification
print(len(ndatasets_t))
#########################################################################################################################
#                                    END OF DATA-PRE-PROCESS PORTION
#########################################################################################################################


#########################################################################################################################
#                                           MODELING PORTION
#########################################################################################################################

# Set the number of time steps in weather data
time_steps = 12
# Set the validation factor (since there is much less validation sets than training sets)
validation_scale = float(16) / float(4)

# Define the number of hidden nodes in Graph Model
hSize = 40

# Initialize the model
model = GATRNN(num_node_static_features=3, num_edge_static_features=3,num_node_dynamic_features=1,num_edge_dynamic_features=1, hidden_size=hSize)

# Move the model to the GPU
model.to(device)

# Loss function and optimizer
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)#,weight_decay=1e-3,eps=1e-3)  # Adam optimizer

# Create lists to store training and validation loss in 
tLOSS =[]
vLOSS =[]

# Number of epoch's to train model for
epochs =5#1500

# Training loop
for epoch in range(epochs):

    # Create lists for current iteration loss
    training_loss = []
    validation_loss = []
    
    # For one epoch, loop over all training data
    for key in ndatasets_t:
        # Unpack and organize data to be passed to model
        # Also move data to GPU
        node_static_feats = key['node_static_features'].to(device)
        edge_static_feats = key['edge_static_features'].to(device)
        node_dynamic_feats = key['node_dynamic_features'].to(device)
        edge_index = key['edge_index'].to(device)
        targets = key['targets'].to(device)

        # Pass the model and training data to train model function defined in functions.py
        tloss = trainGAT(model, node_static_feats,edge_static_feats, node_dynamic_feats, edge_index, targets, optimizer, criterion, float(sF['rangeImpact']))
        
        # Append the current dataset training loss to iteration loss list
        training_loss.append(tloss)
    
    # Calculate the average loss value over all cases for iteration
    epochTloss = sum(training_loss)/len(training_loss)
    
    # Append the average loss value for iteration to tLOSS
    tLOSS.append(epochTloss)

    # For one epoch, loop over all training data
    for key in ndatasets_v:
        # Unpack and organize data to be passed to model
        # Also move data to GPU
        node_static_feats = key['node_static_features'].to(device)
        edge_static_feats = key['edge_static_features'].to(device)
        node_dynamic_feats = key['node_dynamic_features'].to(device)
        edge_index = key['edge_index'].to(device)
        targets = key['targets'].to(device)
        
        # Pass the model and validation data to validate model function defined in functions.py
        vloss = validGAT(model, node_static_feats,edge_static_feats, node_dynamic_feats, edge_index, targets, optimizer, criterion, float(sF['rangeImpact']))
        
        # Append the current dataset validation loss to iteration loss list
        validation_loss.append(vloss)

    # Calculate the average loss value over all cases for iteration and scale by validation factor due to less number of cases as compared to training
    epochVloss = (sum(validation_loss)/len(validation_loss))*validation_scale
    
    # Append the average loss value for iteration to vLOSS
    vLOSS.append(epochVloss)

    # Provide visual update in console
    print('.', end ="", flush=True)
    
    # Provide model performance update in console
    if (epoch % 100 == 0) or (epoch == epochs-1):
        print(f'\nEpoch {epoch} | Training Loss: {epochTloss:.3} | Validation Loss: {epochVloss:.3} | Train RMSE: {np.sqrt(epochTloss):.3} | Valid RMSE: {np.sqrt(epochVloss):.3}')



# Create an empty RMSE list for Topology A validation Cases
rmseA_list = []

# Set the model to evaluation mode
model.eval()
for n in range(len(datasets_vA)):
  with torch.no_grad():
    # Grab normalized input data from list
    key = ndatasets_vA[n]
    
    # Grab denormalized measured output data from list
    rkey = datasets_vA[n]

    # Assign node, edge, and target data to respective variables
    node_static_feats = key['node_static_features'].to(device)
    edge_static_feats = key['edge_static_features'].to(device)
    node_dynamic_feats = key['node_dynamic_features'].to(device)
    edge_index = key['edge_index'].to(device)
    targets = rkey['targets'].to(device)

    # Pass the input data to model to get estimated output
    final_predictions = model(node_static_feats, edge_static_feats, node_dynamic_feats, edge_index)   
    
    # Convert the measured target tensor to a numpy array 
    targ =targets.detach().cpu().numpy()

    # Convert the model estimated target tensor to a numpy array 
    estimated =torch.transpose(final_predictions, 1, 0)[0,:].detach().cpu().numpy()
    
    # Denormalize the model's estimated output
    actualEstimated  = deNormalize(estimated,float(sF['meanImpact']),float(sF['rangeImpact']))

    # Calculate the mean between measuted and estimated, while taking into account the base (or max impact score) to scale by
    mseA = np.mean(((targ.flatten() - actualEstimated.flatten())/float(sF['rangeImpact']))**2)
    
    # Take the square root of the mean to get RMSE
    rmseA = np.sqrt(mseA)
    
    # Append the RMSE to the RMSE list for Topology A
    rmseA_list.append(rmseA)
    
    # Print the RMSE error for the current Topology A validation case  
    print('RMSE (A Cases) = ' +str(rmseA))

    # Calculate the absolute difference between the measured and estimated values
    error = np.abs(targ.flatten()-actualEstimated.flatten())
    
    # Create an array of time values to plot the weather data against
    time = np.linspace(1,12,12)
    
    # Convert weather data to numpy array 
    input_weather = node_dynamic_feats.detach().cpu().numpy()[0,:]

# Take the mean of all the RMSE values for Topology A validation cases
average_rmseA = np.mean(rmseA_list)

# Print the average RMSE over all Topology A validation cases
print('Model RMSE for All A Cases = ' + str(average_rmseA))




# Create an empty RMSE list for Topology B validation Cases
rmseB_list = []

# Set the model to evaluation mode
model.eval()

# Loop through validation B cases
for n in range(len(datasets_vB)):
    
    with torch.no_grad():
        # Grab normalized measured output data from list
        key = ndatasets_vB[n]
        # Grab denormalized measured output data from list
        rkey = datasets_vB[n]
        
        # Assign node, edge, and target data to respective variables
        node_static_feats = key['node_static_features'].to(device)
        edge_static_feats = key['edge_static_features'].to(device)
        node_dynamic_feats = key['node_dynamic_features'].to(device)
        edge_index = key['edge_index'].to(device)
        targets = rkey['targets'].to(device)

        # Pass the input data to model to get estimated output
        final_predictions = model(node_static_feats, edge_static_feats, node_dynamic_feats, edge_index)   
        
        # Convert the measured target tensor to a numpy array 
        targ =targets.detach().cpu().numpy()

        # Convert the model estimated target tensor to a numpy array 
        estimated =torch.transpose(final_predictions, 1, 0)[0,:].detach().cpu().numpy()
        
        # Denormalize the model's estimated output
        actualEstimated  = deNormalize(estimated,float(sF['meanImpact']),float(sF['rangeImpact']))

        # Calculate the mean between measuted and estimated, while taking into account the base (or max impact score) to scale by
        mseB = np.mean(((targ.flatten() - actualEstimated.flatten())/float(sF['rangeImpact']))**2)   
        
        # Take the square root of the mean to get RMSE
        rmseB = np.sqrt(mseB)
        
        # Append the RMSE to the RMSE list for Topology B
        rmseB_list.append(rmseB)
        
        # Print the RMSE error for the current Topology A validation case  
        print('RMSE (B Case) = ' +str(rmseB))

        # Calculate the absolute difference between the measured and estimated values
        error = np.abs(targ.flatten()-actualEstimated.flatten())
        
        # Create an array of time values to plot the weather data against
        time = np.linspace(1,12,12)
        
        # Convert weather data to numpy array 
        input_weather = node_dynamic_feats.detach().cpu().numpy()[0,:]

# Take the mean of all the RMSE values for Topology B validation cases
average_rmseB = np.mean(rmseB_list)

# Print the average RMSE over all Topology B validation cases
print('Model RMSE for B Cases = ' + str(average_rmseB))






# Create an empty RMSE list for Topology A and Topology B Validation Cases
rmseAB_list = []

# Set the model to evaluation mode
model.eval()

# Loop through combined validation cases  
for n in range(len(datasets_v)):
    with torch.no_grad():
        
        # Create a figure for each validationc case to plot
        fig,ax = plt.subplots(3,1,constrained_layout=True)
        
        # Grab normalized measured output data from list
        key = ndatasets_v[n]
        
        # Grab denormalized measured output data from list
        rkey = datasets_v[n]

        # Assign node, edge, and target data to respective variables
        node_static_feats = key['node_static_features'].to(device)
        edge_static_feats = key['edge_static_features'].to(device)
        node_dynamic_feats = key['node_dynamic_features'].to(device)
        edge_index = key['edge_index'].to(device)
        targets = rkey['targets'].to(device)

        # Pass the input data to model to get estimated output
        final_predictions = model(node_static_feats, edge_static_feats, node_dynamic_feats, edge_index)

        # Convert the measured target tensor to a numpy array    
        targ =targets.detach().cpu().numpy()

        # Convert the model estimated target tensor to a numpy array 
        estimated =torch.transpose(final_predictions, 1, 0)[0,:].detach().cpu().numpy()
        
        # Denormalize the estimated model output
        actualEstimated  = deNormalize(estimated,float(sF['meanImpact']),float(sF['rangeImpact']))

        # Calculate the mean between measuted and estimated, while taking into account the base (or max impact score) to scale by
        mseAB = np.mean(((targ.flatten() - actualEstimated.flatten())/float(sF['rangeImpact']))**2)

        # Calculate the square root of the mean squared error
        rmseAB = np.sqrt(mseAB)
        
        # Append the RMSE to the total validation rmse list
        rmseAB_list.append(rmseAB)

        # Print the RMSE of the current validation case
        print('RMSE (A and B Cases) = ' +str(rmseAB))

        # Calculate the absolute difference between the measured and estimated values
        error = np.abs(targ.flatten()-actualEstimated.flatten())
        
        # Create an array of time values to plot the weather data against
        time = np.linspace(1,12,12)
        
        # Convert weather data to numpy array 
        input_weather = node_dynamic_feats.detach().cpu().numpy()[0,:]

        # Create am array corresponding to the node numbers
        t = np.linspace(1,8,8)
        
        # Plot the wind data for the validation case
        ax[0].plot(time, deNormalize(input_weather,float(sF['meanSpeed']),float(sF['rangeSpeed'])))
        ax[0].set_title('Input Wind Data')
        ax[0].set_xlabel('Time (hr)')
        ax[0].set_ylabel('Wind Speed (mph)')
        
        # Print the measured vs estimated output
        ax[1].scatter(t,targ,label='Actual')
        ax[1].scatter(t,actualEstimated,label='Model')
        ax[1].set_xlabel('Node #')
        ax[1].set_title('Predicted vs Measured Impact')
        ax[1].set_ylabel('Impact Level')
        ax[1].legend()

        # Print the absolute error between measured and estimated
        ax[2].set_title('Predicted vs Measured Error')
        ax[2].scatter(t,error)
        ax[2].set_xlabel('Node #')
        ax[2].set_ylabel('(Abs) Difference')
        
        # Set the title of the figure to include the RMSE errors
        fig.suptitle(f'Topology {key["topology"]} : Scenario {key["scenario"]} : RMSE={rmseAB:.3f}')
    


# Take the mean of all the RMSE values for combined validation cases
average_rmseAB = np.mean(rmseAB_list)

# Print the average RMSE over all Topology B validation cases
print('Model RMSE for All Cases = ' + str(average_rmseAB))

# Plot the training and validatiopn loss vs the number of epochs 
fig = plt.figure()
plt.plot(np.sqrt(np.asarray(tLOSS)),label='train')
plt.plot(np.sqrt(np.asarray(vLOSS)),label='valid')
plt.xlabel('Epoch #')
plt.ylabel('Loss (RMSE)')
plt.title('Model Training Loss')
plt.legend()
plt.show()




