# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:45:35 2021

@author: lxq
"""
import torch
from torch.utils.data import DataLoader
#import pickle
import torch.nn as nn
import tensorflow as tf
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from sklearn.neural_network import MLPClassifier
from rdkit.Chem import AllChem
from rdkit import rdBase
#from tqdm import tqdm
import time
from datetime import timedelta
import joblib
from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
import os
rdBase.DisableLog('rdApp.error')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
start_time = time.time() 

restore_from = '/home/stt/XL/lxq/data/chembl28/OX1R/CKPT/same_last.ckpt'
# Read vocabulary from a file
voc = Vocabulary(init_from_file="/home/stt/XL/lxq/data/chembl28/chembl_28_voc")

# Create a Dataset from a SMILES file
moldata = MolData("/home/stt/XL/lxq/data/chembl28/8645.smi", voc)
data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

Prior = RNN(voc)

 # Can restore from a saved RNN
if restore_from:
    Prior.rnn.load_state_dict(torch.load(restore_from)) 
#for param in rior.rnn.parameters():
#    param.requires_grad = False

#Prior.rnn.embedding.weight.requires_grad = False
#Prior.rnn.linear = nn.Linear(512, voc.vocab_size).cuda()
  
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Prior.rnn.parameters()), lr = 0.0001)

optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.0001) 


def mlp_pred(smiles):
    mlp_model = joblib.load('/home/stt/XL/lxq/data/chembl28/OX1R/Topo-mlp.model') 
    X_test=[]
    for i,smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
#       fp = AllChem.GetMACCSKeysFingerprint(mol,167) #MACCS
#       fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)  #Morgan
        fp = Chem.RDKFingerprint(mol)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp,arr)
        X_test.append(arr)
    X_test = np.asarray(X_test)
    predict = mlp_model.predict(X_test) 
    num_ac=0
    for pre in predict:
        if pre==1:
            num_ac+=1
    return num_ac


total_loss=[]
total_valipro=[]
#total_actpro_cnn=[]
total_actpro_svm=[]
max_act_pro=0
for epoch in range(1,201):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
#    if epoch!=0:
#        decrease_learning_rate(optimizer, decrease_by=0.01)

    for step, batch in enumerate(data):

            # Sample from DataLoader
        seqs = batch.long()

            # Calculate loss
        log_p, _ = Prior.likelihood(seqs)
        loss = - log_p.mean()

            # Calculate gradients and take a step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if  step % 8 == 0:
#            decrease_learning_rate(optimizer, decrease_by=0.03)
#                tqdm.write("*" * 50)
            print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss))
            total_loss.append(float(loss))
            seqs, likelihood, _ = Prior.sample(128)
            valid = 0
#            smiles=[]
            vali_smi=[]
            for i, seq in enumerate(seqs.cpu().numpy()):
                smile = voc.decode(seq)
#                smiles.append(smile)
                if Chem.MolFromSmiles(smile):
                    valid += 1
                    vali_smi.append(smile)
                if i < 5:
                    print(smile)
           
            val_pro = valid / len(seqs)
            act_num_svm = mlp_pred(vali_smi)
            act_pro_svm = act_num_svm / len(seqs)
            total_valipro.append(val_pro)
            total_actpro_svm.append(act_pro_svm)
            print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
            print("\n{:>4.1f}% svm  activate SMILES".format(100 * act_num_svm / len(seqs)))
            print("*" * 50 + "\n")
            if act_pro_svm > max_act_pro:
                max_act_pro=act_pro_svm
                torch.save(Prior.rnn.state_dict(), "/home/stt/XL/lxq/data/chembl28/OX1R/CKPT/best-2.ckpt")

        # Save the Prior
    torch.save(Prior.rnn.state_dict(), "/home/stt/XL/lxq/data/chembl28/OX1R/CKPT/final-2.ckpt")
end_time = time.time()
time_diff = end_time - start_time    # Compute the total time.
print('Time usage:' + str(timedelta(seconds= int(round(time_diff)))) )
print(total_loss)
print('############  total_valid_pro #############')
print(total_valipro)
print('############# total_act_svm_pro #############')
print(total_actpro_svm)
