# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 13:56:43 2021

@author: lxq
"""

from data_structs import construct_vocabulary


fname = 'D:/DAP/data/chembl28.smi'
voc_path = 'D:/DAP/data/chembl28_voc'  

smiles_list = []
with open(fname, 'r') as f:
    for line in f:
        smiles_list.append(line.split()[0])


all_chars = construct_vocabulary(smiles_list, voc_path) 
