#from functions import train, validate, hierarchy_pos
import torch
from torch_geometric.nn import GCNConv, GATConv,GATv2Conv
import torch.nn as nn
import numpy as np

torch.manual_seed(0)



class GATRNN(nn.Module):
    def __init__(self, num_node_static_features,num_edge_static_features, num_node_dynamic_features,num_edge_dynamic_features, hidden_size):
        super(GATRNN, self).__init__()
        self.gat_conv = GATv2Conv(num_node_static_features, hidden_size, edge_dim=num_edge_static_features)
        #self.gnn = GCNConv(num_static_features, hidden_size)
        self.lstm = nn.LSTM(hidden_size + num_node_dynamic_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, node_static_features, edge_static_features, node_dynamic_features, edge_index):
        
        time_steps = node_dynamic_features.size(1)
        gat_output = self.gat_conv(node_static_features[:,0,:],edge_index,edge_static_features[:,0,:])

        gat_output = torch.relu(gat_output)
        gat_output = gat_output.unsqueeze(1).expand(-1, time_steps, -1)
        rnn_input = torch.cat((gat_output, node_dynamic_features), dim=2)
        lstm_output, _ = self.lstm(rnn_input)
        output = self.fc(lstm_output)

        # Aggregate across time steps (mean pooling)
        mean_pooled_output = torch.mean(output, dim=1)
       
        return mean_pooled_output