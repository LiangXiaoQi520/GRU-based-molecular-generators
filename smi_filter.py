# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:10:35 2021

@author: lxq
"""


from data_structs import filter_mol
from rdkit import Chem


smiles_fname = '/home/stt/XL/lxq/data/chembl28/chembl_28_chemreps.smi'  
smi_savef = '/home/stt/XL/lxq/data/chembl28/chembl_smi_filter.smi' 
smiles_list = []
with open(smiles_fname, 'r') as f:
    for line in f:
        smiles = line.split(" ")[0]
        mol = Chem.MolFromSmiles(smiles) 
        if filter_mol(mol):
            smiles_list.append(Chem.MolToSmiles(mol)) 



smiles_list = set(smiles_list)


with open(smi_savef, 'w') as f:
    for smiles in smiles_list:
        f.write(smiles + "\n")        
 



           
