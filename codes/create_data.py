import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *

def atom_features(atom, explicit_H = False, use_chirality=False):
    results = one_of_k_encoding_unk(
      atom.GetSymbol(), #11
      [
        'C',
        'N',
        'O',
        'F',
        'P',
        'S',
        'Si'
        'Cl',
        'Br',
        'I',
        'other'
      ]) + one_of_k_encoding(atom.GetDegree(),
                             [0,1, 2, 3, 4 ,5]) + \
              one_of_k_encoding_unk(atom.GetFormalCharge(),[-1,0,1])+ one_of_k_encoding(atom.GetExplicitValence() ,[0,1,2,3,4,5,6]) + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3]) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4 ])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)
    

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size,features,edge_index


compound_iso_smiles = []
opts = ['train','test']
for opt in opts:
        df = pd.read_csv('cyp_data/cyp'+ '_' + opt + '.csv')
        compound_iso_smiles += list( df['compound'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile) 
    smile_graph[smile] = g

# convert to PyTorch data format
processed_data_file_train = 'data/processed/cyp_train.pt'
processed_data_file_test = 'data/processed/cyp_test.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    df = pd.read_csv('cyp_data/cyp_train.csv')
    train_drugs,  train_Y = list(df['compound']),list(df['score'])
    train_drugs,  train_Y = np.asarray(train_drugs), np.asarray(train_Y)
    df = pd.read_csv('cyp_data/cyp_test.csv')
    test_drugs,test_Y = list(df['compound']),list(df['score'])
    test_drugs, test_Y = np.asarray(test_drugs), np.asarray(test_Y)
    # make data PyTorch Geometric ready
    print('preparing ', 'cyp_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data', dataset='cyp_train', xd=train_drugs, y=train_Y,smile_graph=smile_graph)
    print('preparing ','cyp_test.pt in pytorch format!')
    test_data = TestbedDataset(root='data', dataset='cyp_test', xd=test_drugs, y=test_Y,smile_graph=smile_graph)
    print(processed_data_file_train, ' and ', processed_data_file_test,'have been created')
else:
    print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')        
