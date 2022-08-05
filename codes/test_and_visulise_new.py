import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gat_gcn import GAT_GCN
from utils import *
import dgl
from dgl import DGLGraph
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
from rdkit.Chem import rdDepictor, MolSurf
from rdkit.Chem.Draw import rdMolDraw2D, MolToFile, _moltoimg
from torch_geometric.utils import remove_self_loops
import matplotlib.cm as cm
import matplotlib
from IPython.display import SVG, display
import seaborn as sns
import dgl.function as fn

set_random_seed(15)
TEST_BATCH_SIZE = 1024
model_file_name = '17-model_GAT_GCN_cyp.model'
datasets='cyp'
modeling = [GAT_GCN][int(sys.argv[1])]
model_st = modeling.__name__
dataset = 'cyp'

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output,w = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_preds.numpy().flatten()

test_data = TestbedDataset(root='data', dataset='cyp_test')
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
cuda_name = "cuda:0"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model = modeling().to(device)
model.load_state_dict(torch.load(model_file_name))

P = predicting(model, device, test_loader)
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
predic_value = []
for i in P:
    a = sigmoid(i)
    print(a)
    predic_value.append(a)
test_value = pd.DataFrame({'test_pre':predic_value })
name = '8073_model_'+'value_test'+'.csv'
test_value.to_csv(name,sep=',')


