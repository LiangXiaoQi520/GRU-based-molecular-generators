import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs  #topo指纹
#from rdkit.Chem import MACCSkeys #MACCKEYS指纹
#from rdkit.Chem import AllChem 摩根指纹
#from rdkit.Chem.EState import Fingerprinter
#from rdkit.ML.Descriptors import MoleculeDescriptors
#from rdkit.Chem import Descriptors
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.externals import joblib
import sklearn.tree as sk_tree

data = pd.read_csv('/home/stt/XL/lxq/data/chembl28/OX1R/trans-topo_newmolecule.csv')  #训练集路径
smiles  =data['smiles'].tolist() #将data中的smile列提取，并转换为list格式，命名为smiles
#Label = data['label'].tolist()

X_ECFP=[] #设置一个空数据框，方便后面放数据进去。

for i,smi in enumerate(smiles): #for循环，enumerate是返回数据的索引下标和数据本身，如第一个循环，i=1，smi就是smile数据。
    if Chem.MolFromSmiles(smi):
        mol = Chem.MolFromSmiles(smi) #从smile格式转换为2D结构。
        #fp = MACCSkeys.GenMACCSKeys(mol)
        fp = Chem.RDKFingerprint(mol)
#       fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) #主要改这个句话，将mol转换为摩根指纹，圆形指纹半径为2，nBits为2048，我理解为2048个二进制来代表一个分子。
        arr = np.zeros((1,)) #np.zero设置一个给定形状的由0组成的数组。
        DataStructs.ConvertToNumpyArray(fp,arr) #我理解的是将前面生产的指纹转换为numpy格式/
        X_ECFP.append(arr)
x_ECFP = np.asarray(X_ECFP) #将numpy转换为array
model = joblib.load('/home/stt/XL/lxq/data/chembl28/OX1R/Topo-mlp.model') 
y= model.predict_proba(X_ECFP)[:, 1]
w = pd.DataFrame(y)
w.to_csv('trans_topo__modelpredict.csv')  
