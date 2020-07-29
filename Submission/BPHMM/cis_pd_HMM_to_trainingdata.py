import pandas as pd
import numpy as np
import scipy.io as sio
import pickle
import sys

label_Train = pd.read_csv('../data_labels/CIS-PD_Training_Data_IDs_Labels.csv')
label_Test = pd.read_csv('../data_labels/CIS-PD_Testing_Data_IDs_Labels.csv')


id_list = label_Train['subject_id'].unique()
c_Train = list(label_Train.groupby('subject_id')['measurement_id'].count())
c_Test  = list(label_Test.groupby('subject_id')['measurement_id'].count())



def count(subject_id,data):
    l = []
    if data == 'Train':
        measurement_id = label_Train[label_Train['subject_id'] == subject_id]['measurement_id'].values
        for id in measurement_id:

            m = pd.read_csv('../training_data/'+id+'.csv')
            m['step'] = m['Timestamp'].map(lambda x:int(x*5)/10)
            m = m.groupby('step').mean()
            l.append(len(m))

    else:
        measurement_id = label_Test[label_Test['subject_id'] == subject_id]['measurement_id'].values

        for id in measurement_id:
            m = pd.read_csv('../testing_data/'+id+'.csv')
            m['step'] = m['Timestamp'].map(lambda x:int(x*5)/10)
            m = m.groupby('step').mean()
            l.append(len(m))

    return l


def create_tran(n):
    counter = 0
    #all_tran = {}
    for i,id in enumerate(id_list):
        
        cur = 0
        for data in ["Train","Test"][:n]:
            l = count(id,data)
            for j in l:
                seq = mat["Psi"][0]["stateSeq"][-1]["z"][0][i][0][cur:cur+j]
                cur += j
                cur += 1
                tran = np.zeros((c_state,c_state))

                for ts in range(j-1):
                    tran[seq[ts]-1][seq[ts+1]-1] += 1
                tran = np.where(tran==0,0.1,tran)

                tran = tran/tran.sum(axis = 1)
                all_tran[counter] = tran
                counter += 1
        print(id,"done")

def create_HMM(N,D):
    for i in range(N):
        tran = all_tran[i]
        d,v = np.linalg.eig(tran.T)
        pos_zero = np.argmin(abs(d-1))

        if np.round(d[pos_zero],6) != 1:
            print('{} has no eigenvalue equal to 1'.format(i))

        p = v[pos_zero]
        p /= sum(p)
        all_HMM[i,:D] = p




if __name__ == '__main__':
    i = 0
    while i < len(sys.argv[1:]):
        route = sys.argv[1:][i]
        mat = sio.loadmat('../result/'+str(route)+'/SamplerOutput.mat')
        c_state = (mat["Psi"][0]["F"][-1]).shape[1]
        all_tran = {}
        if sys.argv[1:][i+1] == 'Test':
            create_tran(2)
        else:
            create_tran(1)
            c_Test = [0 for i in c_Test]

        N = len(all_tran)
        D = all_tran[0].shape[1]
        all_HMM = np.zeros((N,2*D))
        create_HMM(N,D)
        print('Total data seq:{}'.format(N))
        print('Length of vectors:{}'.format(2*D))
        test_HMM = np.zeros((1,2*D))
        train_HMM = np.zeros((1,2*D))
        cur = 0
        for l in range(len(c_Train)):
            all_HMM[cur:c_Train[l]+c_Test[l]+cur,D:] = np.tile(all_HMM[cur:c_Train[l]+c_Test[l]+cur,:D].mean(axis = 0),(c_Train[l]+c_Test[l],1))
            train_HMM = np.concatenate((train_HMM,all_HMM[cur:cur+c_Train[l]]))
            cur += c_Train[l]
            test_HMM = np.concatenate((test_HMM,all_HMM[cur:cur+c_Test[l]]))
            cur += c_Test[l]

        with open(str(route).replace('/','_')+'_seq_train_HMM.txt', 'wb') as handle:
            pickle.dump(train_HMM, handle)

        with open(str(route).replace('/','_')+'_seq_test_HMM.txt', 'wb') as handle:
            pickle.dump(test_HMM, handle)


        i += 2
'''
for i in range(len(all_tran)):
    if i % 50 == 0:
        print(i,all_tran[i])
'''
