# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:33:51 2020

@author: eva
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#0:fully on 4:fully 0ff
label = pd.read_csv('data_labels/CIS-PD_Training_Data_IDs_Labels.csv')

step = pd.DataFrame(np.arange(0,1200.1,0.1),columns = ['step'])
step['step'] = step['step'].map(lambda x:str(round(x,1)))
def data_array(subject_id):
    measurement_id = label[label['subject_id'] == subject_id]['measurement_id'].values
    group = []
    for id in measurement_id:
        m = pd.read_csv('training_data/'+id+'.csv')
        m['step'] = m['Timestamp'].map(lambda x:str(round(x,1)))
        m= pd.concat([m,step],sort=False)        
        m_red = m.groupby('step').mean()
        group.append(m_red[["X","Y","Z"]].values)
    return group
l = label[label['subject_id'] == 1004]['on_off'].values
per1004 = data_array(1004)
per = np.array(per1004)







measurement_id = label[label['subject_id'] == 1004]['measurement_id'].values
l = label[label['subject_id'] == 1004].loc[0][2]

m = pd.read_csv('training_data/'+measurement_id[0]+'.csv')
m.iloc[:10]
m.iloc[-10:]
m.Y.sum()

m.plot('Timestamp','X')
m.plot('Timestamp','Y')
m.plot('Timestamp','Z')

m['seconds'] = m['Timestamp'].map(lambda x:int(x*10)/10)

m.iloc[:10].groupby('seconds').mean()
xx= pd.concat([m,b],sort=False)
np.arange(0,1200,0.01)
xx_red = xx.groupby('seconds').mean()

m_red = m.groupby('seconds').mean()
def plot(df):
    fig,ax = plt.subplots()
    ax.plot(df['Timestamp'],df.X,df['Timestamp'],df.Y,df['Timestamp'],df.Z)
    plt.legend(('X','Y','Z'))
    plt.show()

plot(m_red)


def plot_multi(s=0,n=0):
    fig,axs = plt.subplots(n-s,1)
    t = []
    for i in range(n-s):
        measurement_id = label[label['subject_id'] == 1004].loc[s+i][0]
        m = pd.read_csv('training_data/'+measurement_id+'.csv')
        l = label[label['subject_id'] == 1004].loc[i][2]
        m['seconds'] = m['Timestamp'].map(int)+1
        m = m.groupby('seconds').mean()
        t.extend(m[["X","Y","Z"]].values)
        axs[i].plot(m['Timestamp'],m.X,m['Timestamp'],m.Y,m['Timestamp'],m.Z)
        axs[i].set_title("{}:{}".format('label',l))
        #axs[i].legend(('X','Y','Z'))
    
    
    
    fig.tight_layout()
    #return t        
t = plot_multi(4,7)
t = np.array(t)
plt.plot(range(3600),t[:,0],range(3600),t[:,1],range(3600),t[:,2])



threedee = plt.figure().gca(projection='3d')
threedee.plot(m.Timestamp,m.X,m.Y)
threedee.set_xlabel('X')
threedee.set_ylabel('Y')
threedee.set_zlabel('Z')
plt.show()