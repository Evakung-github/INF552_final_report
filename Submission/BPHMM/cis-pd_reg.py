import pandas as pd
import numpy as np
import scipy.io as sio
import pickle
import sklearn.svm as svm
from sklearn.externals import joblib 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,train_test_split,cross_val_score,ShuffleSplit,StratifiedShuffleSplit

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
        
        #Final test its MSE
        
        seed = np.random.randint(100)
        trainX,testX,trainY,testY = train_test_split(X,Y,random_state = 20,test_size = 0.25)
        print("seed: ",seed)

        # Parameters change according to the labels
        lasso = Lasso(alpha=0.005,max_iter=5000)
        lasso.fit(trainX,trainY)
        print("lasso_mse:",np.mean((lasso.predict(testX)-testY)**2))


        #GridSearchCV
        
        parameters = {'alpha':[0.005, 0.01, 0.1, 1, 10]}

        lr = LinearRegression()
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
        
        clf_lasso=GridSearchCV(Lasso(max_iter=5000),parameters,cv=cv)
        clf_ridge=GridSearchCV(Ridge(max_iter=5000),parameters,cv=cv)
        clf_lasso.fit(X,Y)
        clf_ridge.fit(X,Y)      
        
        print(sys.argv[i+2])
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=1)

        scores_accuracy = cross_val_score(clf_lasso.best_estimator_, X, Y, cv=cv,scoring='neg_mean_squared_error')
        print("lasso")
        print(clf_lasso.best_estimator_)
        print(scores_accuracy.mean())
        print(scores_accuracy.std())
        

        scores_accuracy = cross_val_score(clf_ridge.best_estimator_, X, Y, cv=cv,scoring='neg_mean_squared_error')
        print("ridge")
        print(clf_ridge.best_estimator_)
        print(scores_accuracy.mean())
        print(scores_accuracy.std())


        i += 2
        
