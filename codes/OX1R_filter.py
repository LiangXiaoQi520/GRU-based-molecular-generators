# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:37:01 2021

@author: lxq
"""

from data_structs import filter_file_on_chars
from rdkit import Chem
import numpy as np

smiles_fname = '/home/stt/XL/lxq/data/chembl28/com_OX2R_filter.smi' 
voc_fname = '/home/stt/XL/lxq/data/chembl28/chembl_28_voc'  


filter_file_on_chars(smiles_fname, voc_fname)   


