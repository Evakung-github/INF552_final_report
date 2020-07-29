import pandas as pd
import numpy as np
import scipy.io as sio
import pickle
import sklearn.svm as svm
from sklearn.externals import joblib 
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit,StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import sys
import os
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn import metrics



label = pd.read_csv('../data_labels/CIS-PD_Training_Data_IDs_Labels.csv')
#trainY = np.array(label['dyskinesia'])
# 'on_off', 'dyskinesia', 'tremor'

if __name__ == '__main__':
    i = 0
    while i < len(sys.argv[1:]):
        fn = sys.argv[i+1]
        trainY = np.array(label[sys.argv[i+2]])


        with open(fn, 'rb') as handle:
            all_presentations = pickle.loads(handle.read())
        
        filename,ext = os.path.splitext(fn)
        
        not_nan = ~np.isnan(trainY)
        
        X = all_presentations[1:][not_nan]
        Y = trainY[not_nan]
        D = X.shape[1]
        #Test if concatenating mean value within each patient is helpful
        #X = X[:,:(D//2)]
        seed = np.random.randint(100)
        trainX,testX,trainY,testY = train_test_split(X,Y,random_state = 20,test_size = 0.25)
        #LinearSVC doesn't converge
#        print("seed: ",seed)
#        Lsvc = svm.LinearSVC(class_weight='balanced')
#        Lsvc.fit(trainX,trainY)
#        pred_trainY = Lsvc.predict(trainX)
#        pred_testY = Lsvc.predict(testX)

#        print("LSVC_train accuracy rate: {} F1-score: {}".format(Lsvc.score(trainX,trainY),f1_score(trainY,pred_trainY,average = 'weighted')))
#        print("LSVC_test  accuracy rate: {} F1-score: {}".format(Lsvc.score(testX,testY),f1_score(testY,pred_testY,average = 'weighted')))

#        print(metrics.confusion_matrix(trainY, pred_trainY))
#        print(metrics.classification_report(trainY,pred_trainY, digits=3))
#        print(metrics.confusion_matrix(testY, pred_testY))
#        print(metrics.classification_report(testY,pred_testY, digits=3))

        #Final test its MSE
        # Parameters change according to the labels
        
        svc = svm.SVC(class_weight=None,kernel = 'linear',gamma=0.001,degree=3,C=1)
        svc.fit(trainX,trainY)
        pred_trainY = svc.predict(trainX)
        pred_testY = svc.predict(testX)
        print("SVC_train accuracy rate: {} F1-score: {}".format(svc.score(trainX,trainY),f1_score(trainY,pred_trainY,average = 'weighted')))
        print("SVC_test  accuracy rate: {} F1-score: {}".format(svc.score(testX,testY),f1_score(testY,pred_testY,average = 'weighted')))
        print("MSE:{}".format(np.mean((pred_testY-testY)**2)))
        joblib.dump(svc,filename+'_'+sys.argv[i+2]+'_svc.pkl')

        print(metrics.confusion_matrix(trainY, pred_trainY))
        print(metrics.classification_report(trainY,pred_trainY, digits=3))
        print(metrics.confusion_matrix(testY, pred_testY))
        print(metrics.classification_report(testY,pred_testY, digits=3))
        
        #GridSearchCV
        
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
        parameters = {'kernel':('linear', 'rbf','poly'),'degree':[3,5,7], 'C':[0.001, 0.01, 0.1, 1, 10],'gamma' : [0.001, 0.01, 0.1, 1],'class_weight':['balanced',None]}
        clf = GridSearchCV(svm.SVC(max_iter = 5000), parameters,cv=cv)
        clf.fit(X,Y)
        
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=1)
        scores_accuracy = cross_val_score(clf.best_estimator_, X, Y, cv=cv)
        scores_f1 = cross_val_score(clf.best_estimator_,X,Y,cv=cv,scoring='f1_weighted')
        print(sys.argv[i+2])
        print(clf.best_estimator_)
        print("cv mean:",scores_accuracy.mean())
        print("cv std:",scores_accuracy.std())

        i += 2

        
        
        
        
        joblib.dump(clf,filename+'_clf.pkl')
        #joblib.dump(svc, filename+'_svc0410.pkl')



