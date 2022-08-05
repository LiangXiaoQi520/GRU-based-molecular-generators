import numpy as np
import pandas as pd
import sys, os
import joblib
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from utils import *
import matplotlib.pyplot as plt
from torchsampler import ImbalancedDatasetSampler
from sklearn.model_selection import StratifiedKFold
from joblib import dump,load


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_train = torch.Tensor()
    total_label = torch.Tensor()
    train_losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output,w = model(data)
        loss = loss_fn(output, data.y.view(-1,1).float()).to(device)
        loss = torch.mean(loss).float()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))                                                                                                                                 	
    total_train = torch.cat((total_train, output.cpu()), 0)
    total_label = torch.cat((total_label, data.y.view(-1, 1).cpu()), 0)
    G_train = total_label.detach().numpy().flatten()
    P_train = total_train.detach().numpy().flatten()
    ret = [auc(G_train,P_train),pre(G_train,P_train),recall(G_train,P_train),f1(G_train,P_train),acc(G_train,P_train),mcc(G_train,P_train),spe(G_train,P_train)]
    print('train_auc',ret[0])
    print('train_pre',ret[1])
    print('train_recall',ret[2])
    print('train_f1',ret[3])
    print('train_acc',ret[4])
    print('train_mcc',ret[5])
    print('train_spe',ret[6])
    print('train_loss',np.average(train_losses))
    return G_train, P_train, np.average(train_losses)

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    losses = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output,w = model(data)
            loss = loss_fn(output, data.y.view(-1,1).float())
            loss = torch.mean(loss).float().to(device)
            losses.append(loss.item())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten(),np.average(losses)



modeling = [GATNet, GAT_GCN][int(sys.argv[1])]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)
    
TRAIN_BATCH_SIZE = 1024
TEST_BATCH_SIZE = 1024
LR = 0.001
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

processed_data_file_train = 'data/processed/cyp_train.pt'
processed_data_file_test = 'data/processed/cyp_test.pt'

if  (not os.path.isfile(processed_data_file_train)):
    print('please run create_data.py to prepare data in pytorch format!')
else:
    train_data = TestbedDataset(root='data', dataset='cyp_train')
    test_data = TestbedDataset(root='data', dataset='cyp_test')
    train_data_y = pd.read_csv('cyp_data/cyp_train.csv')['score']

skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42 )
skf.get_n_splits(train_data,train_data_y) 


train_mccs=[]
train_accs = []
train_aucs=[]
train_pres=[]
train_recalls=[]
train_f1s=[]
train_spes=[]

val_mccs=[]
val_accs = []
val_aucs=[]
val_pres=[]
val_recalls=[]
val_f1s=[]
val_spes=[]

test_mccs=[]
test_accs = []
test_aucs=[]
test_pres=[]
test_recalls=[]
test_f1s=[]
test_spes=[]

valid_num = 0
train_losses = []
val_losses=[]
# Main program: iterate over seven cross validation.
for train_index, val_index in skf.split(train_data,train_data_y):
    valid_num = valid_num+1
    print('\nrunning on GAT_GCN_cyp on {} validation'.format(valid_num))
    x_train_, x_val_ = train_data[train_index], train_data[val_index]
    y_train_, y_val_ = np.array(train_data_y)[train_index], np.array(train_data_y)[val_index]

    lables_unique, counts = np.unique(y_train_,return_counts = True)
    class_weights = [sum(counts)/ c for c in counts]
    example_weights = [class_weights[e] for e in y_train_]
    sampler = WeightedRandomSampler(example_weights, len(y_train_))
    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(x_train_, batch_size=TRAIN_BATCH_SIZE, sampler=sampler)
    valid_loader = DataLoader(x_val_, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.BCELoss()
    #loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np.to(device))
    #loss_fn = torch.nn.BCEWithLogitsLoss()
   
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss = 100
    best_test_auc = 1000
    best_test_ci = 0
    best_epoch = -1
    patience = 30
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    #model_file_name = 'model_GAT_GCN_cyp.model'
    model_file_name = 're_model_'+'GCN'+str(valid_num)+'.model'
    result_file_name = 'result_GAT_GCN_cyp.csv'
    for epoch in range(NUM_EPOCHS):
        G_T,P_T,train_loss = train(model, device, train_loader, optimizer, epoch+1)
        print('predicting for valid data')
        G,P,loss_valid= predicting(model, device, valid_loader )
        loss_valid_value = loss_valid
        print('valid_loss',loss_valid)
        print('valid_auc',auc(G,P))
        print('valid_pre',pre(G,P))
        print('valid_recall',recall(G,P))
        print('valid_f1',f1(G,P))
        print('valid_acc',acc(G,P))
        print('valid_mcc',mcc(G,P))
        print('valid_spe',spe(G,P))
        train_losses.append(np.array(train_loss))
        train_accs.append(acc(G_T,P_T))
        train_aucs.append(auc(G_T,P_T))
        train_pres.append(pre(G_T,P_T))
        train_recalls.append(recall(G_T,P_T))
        train_f1s.append(f1(G_T,P_T))
        train_mccs.append(mcc(G_T,P_T))
        train_spes.append(spe(G_T,P_T))
        val_losses.append(np.array(loss_valid))
        val_accs.append(acc(G,P))
        val_aucs.append(auc(G,P))
        val_pres.append(pre(G,P))
        val_recalls.append(recall(G,P))
        val_f1s.append(f1(G,P))
        val_mccs.append(mcc(G,P))
        val_spes.append(spe(G,P))
        b = pd.DataFrame({'train_loss':train_losses,'valid_loss':val_losses,'train_acc':train_accs,'valid_acc':val_accs})
        names = 'GNN_'+'loss_acc'+str(valid_num)+'.csv'
        b.to_csv(names,sep=',') 
        early_stopping(loss_valid, model)      
        if early_stopping.early_stop:                
            print("Early stopping")              
            torch.save(model.state_dict(), model_file_name)
            #joblib.dump(model, "/home/qmy/lxq/GCN.model")
            print('predicting for test data')
            
            ret_test = [auc(G,P),pre(G,P),recall(G,P),f1(G,P),acc(G,P),mcc(G,P),spe(G,P)]
            print('cyp_test ',best_epoch,'auc',ret_test[0],'pre',ret_test[1],'recall',ret_test[2],'f1',ret_test[3],'acc',ret_test[4],'mcc',ret_test[5],'spe',ret_test[6])
            test_mccs.append(ret_test[5])
            test_accs.append(ret_test[4])
            test_aucs.append(ret_test[0])
            test_pres.append(ret_test[1])
            test_recalls.append(ret_test[2])
            test_f1s.append(ret_test[3])
            test_spes.append(ret_test[6])
            break
        else:
            print('no early stopping')
    
def get_average(list_):
    sum = 0
    for item in list_:
        sum += item
    return sum/len(list_)

print('five_cross_validation results')
print('train_accs:',get_average(train_accs))
print('train_aucs:',get_average(train_aucs))
print('train_sens:',get_average(train_recalls))
print('train_spes:',get_average(train_spes))
print('train_mccs:',get_average(train_mccs))
print('train_f1s:',get_average(train_f1s))
print('train_pres:',get_average(train_pres))

print('val_accs:',get_average(val_accs))
print('val_aucs:',get_average(val_aucs))
print('val_sens:',get_average(val_recalls))
print('val_spes:',get_average(val_spes))
print('val_mccs:',get_average(val_mccs))
print('val_f1s:',get_average(val_f1s))
print('val_pres:',get_average(val_pres))

print('result of cyp_test')
print('test_accs:',get_average(test_accs))
print('test_aucs:',get_average(test_aucs))
print('test_sens:',get_average(test_recalls))
print('test_spes:',get_average(test_spes))
print('test_mccs:',get_average( test_mccs))
print('test_f1s:',get_average(test_f1s))
print('test_pres:',get_average(test_pres))
