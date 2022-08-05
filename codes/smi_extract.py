# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 12:52:41 2021

@author: lxq
"""

import pandas as pd

chembl_path = 'G:\DAP\chembl_28_chemreps.txt'  
save_path = 'G:\DAP\chembl_28_chemreps.smi'         

data = pd.read_table(chembl_path)

smiles_list = data['canonical_smiles'].tolist() 

with open(save_path, 'w') as f:
    for smiles in smiles_list:
        f.write(smiles + "\n")
