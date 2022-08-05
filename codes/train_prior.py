# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:45:35 2022

@author: lxq
"""


import torch
from torch.utils.data import DataLoader
#import pickle
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
#from tqdm import tqdm
import time
from datetime import timedelta
from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

start_time = time.time()

restore_from = None
# Read vocabulary from a file
voc = Vocabulary(init_from_file="/home/stt/XL/lxq/data/chembl28/chembl_28_voc")     

# Create a Dataset from a SMILES file
moldata = MolData("/home/stt/XL/lxq/data/chembl28/chembl.smi", voc)  
data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

Prior = RNN(voc)

 # Can restore from a saved RNN
if restore_from:
    Prior.rnn.load_state_dict(torch.load(restore_from))

optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr = 0.001)  #学习率

total_loss=[]
total_valid = []
max_valid_pro=0
for epoch in range(1, 9):
        # When training on a few million compounds, this model converges
        # in a few of epochs or even faster. If model sized is increased
        # its probably a good idea to check loss against an external set of
        # validation SMILES to make sure we dont overfit too much.
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

            # Every 500 steps we decrease learning rate and print some information
        if step!=0 and step % 200 == 0:
            decrease_learning_rate(optimizer, decrease_by=0.02)    
        if  step % 300 == 0:     
#                tqdm.write("*" * 50)
            print("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss))
            total_loss.append(float(loss))
            seqs, likelihood, _ = Prior.sample(128)
            valid = 0
            for i, seq in enumerate(seqs.cpu().numpy()):
                smile = voc.decode(seq)
#                smiles.append(smile)
                if Chem.MolFromSmiles(smile):
                    valid += 1
#                    vali_smi.append(smile)
                if i < 5:
                    print(smile)
            vali_pro=valid/len(seqs)
            total_valid.append(float(vali_pro))
            print("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
            print("*" * 50 + "\n")

            if vali_pro > max_valid_pro:
                max_valid_pro = vali_pro
                torch.save(Prior.rnn.state_dict(), "/home/stt/XL/lxq/data/chembl28/OX1R/same_best.ckpt")   

        # Save the Prior
    torch.save(Prior.rnn.state_dict(), "/home/stt/XL/lxq/data/chembl28/OX1R/same_last.ckpt")  
end_time = time.time()
time_diff = end_time - start_time    # Compute the total time.
print('Time usage:' + str(timedelta(seconds= int(round(time_diff)))) )
print(total_loss)
print('###############################################')
print(total_valid)

