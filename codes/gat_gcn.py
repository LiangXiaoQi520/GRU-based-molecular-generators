import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils import *

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=37, output_dim=256, dropout=0.6):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=5)
        self.conv2 = GCNConv(num_features_xd*5, num_features_xd*5)
        #self.conv3 = GCNConv(num_features_xd*5, num_features_xd*5)
        #self.conv4 = GCNConv(num_features_xd*5, num_features_xd*5)
        self.fc_g1 = torch.nn.Linear(num_features_xd*5*2,output_dim)
        #self.fc_g1 = torch.nn.Linear(num_features_xd*5*2,1024)
        #self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256,512)
        #self.fc_g3 = nn.Linear(256, 512)
        #self.fc_g4 = nn.Linear(512, 1024)
        #self.fc_g5 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x,w = self.conv1(x, edge_index,return_attention_weights= True)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        #x = self.conv3(x, edge_index)
        #x = self.relu(x)
        #x = self.conv4(x, edge_index)
        #x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc1(x)
        #x = self.fc_g2(x)
        #x = self.fc_g3(x)
        x = self.relu(x)
        x = self.dropout(x)
        out =self.out(x)
        return out,w
