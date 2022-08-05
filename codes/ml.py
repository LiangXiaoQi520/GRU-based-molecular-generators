
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
#from rdkit.Chem.EState import Fingerprinter
#from rdkit.ML.Descriptors import MoleculeDescriptors
#from rdkit.Chem import Descriptors
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
import joblib
import sklearn.tree as sk_tree

data = pd.read_csv('/home/stt/XL/lxq/data/chembl28/OX1R/1-1-training.csv')  
smiles = data['smiles'].tolist()
Label = data['label'].tolist()

X_ECFP=[]

for i,smi in enumerate(smiles):
    if Chem.MolFromSmiles(smi):
#        index_total.append(i)
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp,arr)
        X_ECFP.append(arr)
        
x_ECFP = np.asarray(X_ECFP)


data = pd.read_csv('/home/stt/XL/lxq/data/chembl28/OX1R/1-1-test.csv')  
smiles = data['smiles'].tolist()
Label_t =data['label'].tolist()

X_t=[]

for i,smi in enumerate(smiles):
    if Chem.MolFromSmiles(smi):
#        index_total.append(i)
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
       #fp = MACCSkeys.GenMACCSKeys(mol)
       #fp = Chem.RDKFingerprint(mol)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp,arr)
        X_t.append(arr)
        
x_t = np.asarray(X_t)


skf = StratifiedKFold(n_splits=5,shuffle=True)   
skf.get_n_splits(x_ECFP, Label)       

import sklearn.neural_network as sk_nn
from sklearn import metrics
from sklearn.metrics import confusion_matrix,matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
model =RandomForestClassifier(n_estimators=50, criterion='gini',max_depth=22)  #rf
#model=LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.5, class_weight= 'balanced', random_state=None, solver='liblinear', max_iter=500, multi_class='ovr', verbose=0) #LR 
#model= svm.SVC(C=10,kernel='rbf',gamma=0.0005,probability =True)  #SVM
#model= sk_nn.MLPClassifier(activation='tanh', solver='adam', hidden_layer_sizes=(512,128),alpha=8,learning_rate='adaptive',learning_rate_init=0.001,max_iter=500) #MLP

train_mcc=[]
train_acc = []
train_auc=[]
train_sen=[]
train_spe=[]
val_mcc=[]
val_acc = []
val_auc=[]
val_sen=[]
val_spe=[]
for train_index, val_index in skf.split(x_ECFP, Label):
    x_train_, x_val_ = x_ECFP[train_index], x_ECFP[val_index]
    y_train_, y_val_ = np.array(Label)[train_index], np.array(Label)[val_index]
    
    model.fit(x_train_, y_train_)
    train_ac = model.score(x_train_, y_train_)
    ##计算auc
    fpr,tpr,thresholds=metrics.roc_curve(y_train_, model.predict_proba(x_train_)[:,1])
    train_au=metrics.auc(fpr,tpr)
    ##计算se ,sp
    train_pred = model.predict(x_train_)
    tn, fp, fn, tp = confusion_matrix(y_train_, train_pred).ravel()
    se = tp/(tp+fn)
    sp = tn/(tn+fp)
    mcc = matthews_corrcoef(y_train_, train_pred)
    train_sen.append(se)
    train_spe.append(sp)
    train_acc.append(train_ac)
    train_auc.append(train_au)
    train_mcc.append(mcc)
    
    val_ac=model.score(x_val_, y_val_)
    fpr,tpr,thresholds=metrics.roc_curve(y_val_, model.predict_proba(x_val_)[:,1])
    val_au=metrics.auc(fpr,tpr)

    val_pred = model.predict(x_val_)
    tn, fp, fn, tp = confusion_matrix(y_val_, val_pred).ravel()
    val_se = tp/(tp+fn)
    val_sp = tn/(tn+fp)
    val_mc = matthews_corrcoef(y_val_, val_pred)
    val_sen.append(val_se)
    val_spe.append(val_sp)
    val_acc.append(val_ac)
    val_auc.append(val_au)
    val_mcc.append(val_mc)
    print(val_ac,val_au,val_se, val_sp,val_mc)

def get_average(list_):
    sum = 0
    for item in list_:
        sum += item
    return sum/len(list_)


print('train_acc:',get_average(train_acc))
print('train_auc:',get_average(train_auc))
print('train_sen:',get_average(train_sen))
print('train_spe:',get_average(train_spe))
print('train_mcc:',get_average(train_mcc))


print('val_acc:',get_average(val_acc))
print('val_auc:',get_average(val_auc))
print('val_sen:',get_average(val_sen))
print('val_spe:',get_average(val_spe))
print('val_mcc:',get_average(val_mcc))


print('test_acc:',model.score(x_t,Label_t))
fpr,tpr,thresholds=metrics.roc_curve(Label_t, model.predict_proba(x_t)[:,1])
print('test_auc:',metrics.auc(fpr,tpr))
test_pred = model.predict(x_t)
tn, fp, fn, tp = confusion_matrix(Label_t, test_pred).ravel()
test_se = tp/(tp+fn)
test_sp = tn/(tn+fp)
print('test_se:',test_se)
print('test_sp:',test_sp)
print('test_mcc:', matthews_corrcoef(Label_t,test_pred))
joblib.dump(model,"/home/stt/XL/lxq/data/chembl28/OX1R/morgan-rf.model") 
