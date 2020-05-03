# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:06:43 2019

@author: mingwei
"""

# 0 Import packages
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import data_help
import pickle
N_FOLDS = 10


def extract_pd(test_x,prediction):    
    result = pd.DataFrame(prediction,index = test_x.index, columns = ["SalePrice"])
    return result
    

# N_FOLDS determines the outcome of this function
def produce_holdout(model,train_x,train_y,features):
    train_x = train_x.loc[:,features]
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
    
    return result


if __name__=='__main__':

    # 1 Load data
    train_x = data_help.train_x
    test_x = data_help.test_x
    train_y = data_help.train_y
    
    
    # 2 Start output results
    key_list = ["lightgbm","ridge","svr","xgb"]
    holdout_collect = []
    for key in key_list:
        with open("tunned_models/model_"+key+".pkl","rb") as f:
            my_model = pickle.load(f)
        model = my_model["final_model"]
        features = my_model["final_test_x"].columns
        holdout_collect.append(produce_holdout(model,train_x,train_y,features).values.reshape(-1))
    holdout_collect = np.column_stack(holdout_collect)
    target = train_y.values.reshape(-1)
    
    
    # 3 Start convex optimization
    from cvxopt import solvers
    from cvxopt import matrix
    
    num = holdout_collect.shape[1]
    P_mat = matrix(2*np.dot(holdout_collect.T,holdout_collect))
    q_mat = matrix(-2*np.dot(target.T,holdout_collect))
    G_mat = matrix(np.concatenate((np.eye(num),-np.eye(num)),axis = 0))
    h_mat = matrix(np.concatenate((np.ones(num),np.zeros(num)),axis = 0))   
    b_mat = matrix(1.)
    A_mat = matrix(np.ones(num)).T
    solution = solvers.qp(P_mat,q_mat,G_mat,h_mat,A_mat,b_mat)
    final_weight = np.array(solution["x"]).reshape(-1)
    
    
    # 4 Test holdout prediction
    final_holdout = np.sum(holdout_collect*final_weight,axis = 1)
    holdout_err = np.sqrt(mean_squared_error(train_y,final_holdout))
    print(holdout_err)
    
    
    # 5 Produce weighted prediction, fitting and residual
    key_list = ["lightgbm","ridge","svr","xgb"]
    prediction_collect = [] 
    fitting_collect = []    
    for key in key_list:
        with open("tunned_models/model_"+key+".pkl","rb") as f:
            my_model = pickle.load(f)   
        model = my_model["final_model"]
        features = my_model["final_test_x"].columns   
        model.fit(train_x.loc[:,features],train_y)
        prediction = model.predict(test_x.loc[:,features])
        fitting = model.predict(train_x.loc[:,features])
        prediction_collect.append(prediction)
        fitting_collect.append(fitting)
    fitting_collect = np.column_stack(fitting_collect)
    prediction_collect = np.column_stack(prediction_collect)
    
    final_fitting = np.sum(fitting_collect*final_weight,axis = 1)
    final_prediction = np.sum(prediction_collect*final_weight,axis = 1)
    final_prediction = np.expm1(final_prediction)
    result = extract_pd(test_x,final_prediction)
    result.to_csv("submissions/stacking_0.csv")
    
    final_residual = train_y.values.reshape(-1)-final_fitting
    residual = pd.DataFrame(final_residual, index = train_y.index, columns = ["residual_0"])
    residual.to_csv("original data cache/residual_0.csv")
    
        
  