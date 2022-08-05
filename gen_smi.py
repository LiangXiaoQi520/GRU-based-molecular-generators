# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:58:56 2021

@author: lxq
"""
import torch
from data_structs import Vocabulary
from model import RNN
from rdkit import Chem
from rdkit import rdBase
import pandas as pd
import gc
rdBase.DisableLog('rdApp.error')

restore_from = '/home/stt/XL/lxq/data/chembl28/OX1R/CKPT/final-4.ckpt'   
# Read vocabulary from a file
voc = Vocabulary(init_from_file="/home/stt/XL/lxq/data/chembl28/chembl_28_voc")  
# Create a Dataset from a SMILES file
for n in range(1,2):
    Prior = RNN(voc)
    
     # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from,map_location='cpu'))

    seqs, likelihood, _ = Prior.sample(1000)  
    #seqs_1, likelihood, _ = Prior.sample(1000)
    #seqs_2, likelihood, _ = Prior.sample(1000)
    #seqs_3, likelihood, _ = Prior.sample(1000)
    #seqs_4, likelihood, _ = Prior.sample(1000)
    #seqs_5, likelihood, _ = Prior.sample(1000)
    
    valid = 0
    smiles=[]
    val_smi=[]
    for i, seq in enumerate(seqs.cpu().numpy()):
        smile = voc.decode(seq)
        smiles.append(smile)
        if Chem.MolFromSmiles(smile):
            valid += 1
            val_smi.append(smile)
        if i < 5:
            print(smile)
    
    
    Val_s = pd.DataFrame(data=val_smi,columns=['smiles'])
    Val_s.to_csv('data_gen_500.csv',index=False)   


    print(valid)
    gc.collect()
