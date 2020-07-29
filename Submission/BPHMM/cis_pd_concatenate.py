import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import scipy.io as sio


label_Train = pd.read_csv('data_labels/CIS-PD_Training_Data_IDs_Labels.csv')
label_Test  = pd.read_csv('data_labels/CIS-PD_Testing_Data_IDs_Labels.csv') 


#Return a single wo minutes data
def id_single_data(subject_id,i):
    measurement_id = label_Train[label_Train['subject_id'] == subject_id]['measurement_id'].values
    id = measurement_id[i]
    m = pd.read_csv('training_data/'+id+'.csv')
    m['step'] = m['Timestamp'].map(lambda x:int(x*5)/10)
    m = m.groupby('step').mean()
    return m[["X","Y","Z"]].values


#Return an array of each subject_id with or without scaling within each data points
#seperate data by [0,0,0]
def concatData(subject_id,z = False):
    measurement_id = label_Train[label_Train['subject_id'] == subject_id]['measurement_id'].values
    group = []
    for id in measurement_id:
        m = pd.read_csv('training_data/'+id+'.csv')
        m['step'] = m['Timestamp'].map(lambda x:int(x*5)/10)
        m = m.groupby('step').mean()
        if not z:
            group.extend(m[["X","Y","Z"]].values)
        else:
            group.extend(scale(m[["X","Y","Z"]].values,axis = 0))
        group.append([0,0,0])
    
    measurement_id = label_Test[label_Test['subject_id'] == subject_id]['measurement_id'].values
    
    for id in measurement_id:
        m = pd.read_csv('testing_data/'+id+'.csv')
        m['step'] = m['Timestamp'].map(lambda x:int(x*5)/10)
        m = m.groupby('step').mean()
        if not z:
            group.extend(m[["X","Y","Z"]].values)
        else:
            group.extend(scale(m[["X","Y","Z"]].values,axis = 0))
        group.append([0,0,0])


    return np.array(group)

#Return an array of each subject with scaling within each person
#First scaling, then add [0,0,0] to split the data
def zscoreAmongperson(subject_id):
    measurement_id = label_Train[label_Train['subject_id'] == subject_id]['measurement_id'].values
    l = []
    group = []
    for id in measurement_id:
        m = pd.read_csv('training_data/'+id+'.csv')
        m['step'] = m['Timestamp'].map(lambda x:int(x*5)/10)
        m = m.groupby('step').mean()
        l.append(len(m))
        group.extend(m[["X","Y","Z"]].values)
        #group.append([0,0,0])
    measurement_id = label_Test[label_Test['subject_id'] == subject_id]['measurement_id'].values
   
    for id in measurement_id:
        m = pd.read_csv('testing_data/'+id+'.csv')
        m['step'] = m['Timestamp'].map(lambda x:int(x*5)/10)
        m = m.groupby('step').mean()
        l.append(len(m))
        group.extend(m[["X","Y","Z"]].values)
        #group.append([0,0,0])
    
    group = scale(np.array(group),axis = 0)
    cur = 0
    for i in l:
        cur += i
        group = np.insert(group,cur,np.array([[0,0,0]]),axis = 0)
        cur += 1
    
    return np.array(group)

def createArray(l):
    noScale = []
    Scaledatapoints = []
    Scaleperson = []

    for i,id in enumerate(l):

        noScale.append(concatData(id,False))
        np.savetxt('outputA_Total/'+str(id_list[i])+'.csv',noScale[-1])
        #sio.savemat('outputA_Total/'+str(id_list[i])+'.mat',{'data':noScale[-1]})
        Scaledatapoints.append(concatData(id,True))
        #sio.savemat('outputB_Total/'+str(id_list[i])+'.mat',{'data':Scaledatapoints[-1]})
        np.savetxt('outputB_Total/'+str(id_list[i])+'.csv',Scaledatapoints[-1])
        Scaleperson.append(zscoreAmongperson(id))
        #sio.savemat('outputC_Total/'+str(id_list[i])+'.mat',{'data':Scaleperson[-1]})
        np.savetxt('outputC_Total/'+str(id_list[i])+'.csv',Scaleperson[-1])
        print(id," done!")

    return np.array(noScale),np.array(Scaledatapoints),np.array(Scaleperson)



id_list = label_Train['subject_id'].unique()

A,B,C = createArray(id_list)

#np.savetxt(str(id_list[0])+'.csv',A[0])
#np.savetxt(str(id_list[0])+'_B.csv',B[0])
#np.savetxt(str(id_list[0])+'_C.csv',C[0])
#
#
#sio.savemat(str(id_list[0])+'_B.mat', {'mydata': A[0]})
#
#A[0][:10]
#B[0][:10]
#C[0].sum(axis = 0)
