# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:06:43 2019

@author: mingwei
"""

# 0 Import packages
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import data_help
import pickle
from sklearn.model_selection import KFold
N_FOLDS = 10

def extract_pd(test_x,prediction):    
    result = pd.DataFrame(prediction,index = test_x.index, columns = ["SalePrice"])
    return result


def produce_holdout(model,train_x,train_y):

    train_x = pd.DataFrame(train_x) 
    train_y = pd.DataFrame(train_y)
        
    kf = KFold(N_FOLDS, shuffle=True, random_state=42)
    kf.get_n_splits(train_x,train_y)   
    result = pd.DataFrame(np.zeros(train_x.shape[0]), index=train_x.index, columns=["predictive_price"])
    

    for train_index, test_index in kf.split(train_x, train_y):
        tem_train_x, tem_train_y = train_x.iloc[train_index,:], train_y.iloc[train_index,:]
        tem_test_x, tem_test_y = train_x.iloc[test_index,:], train_y.iloc[test_index,:] 
        tem_train_y = tem_train_y.values.reshape(-1)           
        model.fit(tem_train_x,tem_train_y)
        prediction = model.predict(tem_test_x)
        result.iloc[test_index,:] = prediction.reshape(-1,1)
    
    model.fit(train_x,train_y)
    fit_err = np.sqrt(mean_squared_error(train_y,model.predict(train_x)))    
    return result,fit_err


def test_feature(model,train_x,train_y):
    
    holdout,fit_err = produce_holdout(model,train_x,train_y)
    err = np.sqrt(mean_squared_error(train_y,holdout))
    penalty = err-fit_err
    result = err+penalty if penalty>0 else 100
    
    return result,penalty
  

if __name__=='__main__':

    # 1 Load data
    train_x = data_help.train_x
    test_x = data_help.test_x
    train_y = data_help.train_y
    
    
    # 2 Start output results
    key_list = ["lgb","xgb","robust_lgb"]
    for key in key_list:
        with open("models/"+key+".pkl","rb") as f:
            my_model = pickle.load(f)
        model_type = my_model[-1]
        feature = my_model[-3][1]
        para = my_model[-3][2]
        model = model_type(**para)
        model.fit(train_x.loc[:,feature],train_y)
    
        prediction = np.expm1(model.predict(test_x.loc[:,feature]))
        fit_err = mean_squared_error(model.predict(train_x.loc[:,feature]),train_y.values.reshape(-1))
        fit_err = np.sqrt(fit_err)
        result = extract_pd(test_x,prediction)
        result.to_csv("submissions/tunned_singlepre_"+key+".csv")
        
        print("="*60)
        print("The cv score of model and feature is:")
        print(test_feature(model,train_x.loc[:,feature],train_y))
        print("The type of model is:")
        print(key)
        print("The fitt error is:")
        print(fit_err)
        print("Evaluation on the test data:")
        print(data_help.evaluate(result))
        print("features are:")
        print(feature)
        print("="*60)       

  