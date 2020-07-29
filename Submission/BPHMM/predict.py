# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:25:12 2020

@author: eva
"""

import pandas as pd
import numpy as np
import scipy.io as sio
import pickle
import sys
from sklearn.externals import joblib 
import os

label_Train = pd.read_csv('data_labels/CIS-PD_Training_Data_IDs_Labels.csv')
label_Test = pd.read_csv('data_labels/CIS-PD_Testing_Data_IDs_Labels.csv')


id_list = label_Train['subject_id'].unique()
c_Train = list(label_Train.groupby('subject_id')['measurement_id'].count())
c_Test  = list(label_Test.groupby('subject_id')['measurement_id'].count())


if __name__ == '__main__':
    fn = sys.argv[1]
    filename,ext = os.path.splitext(fn)
    with open(fn, 'rb') as handle:
        testing_data = pickle.loads(handle.read())
    
    model = joblib.load(sys.argv[2])
    
    a = model.predict(testing_data)
    
    with open("dyskinesia.txt",'wb') as fn:
        pickle.dump(a,fn)


