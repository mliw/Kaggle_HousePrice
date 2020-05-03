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


def extract_pd(test_x,prediction):    
    result = pd.DataFrame(prediction,index = test_x.index, columns = ["SalePrice"])
    return result
    

if __name__=='__main__':

    # 1 Load data
    train_x = data_help.train_x
    test_x = data_help.test_x
    train_y = data_help.train_y
    
    
    # 2 Start output results
    key_list = ["lightgbm","ridge","svr","xgb"]
    for key in key_list:
        with open("tunned_models/model_"+key+".pkl","rb") as f:
            my_model = pickle.load(f)
        model = my_model["final_model"]
        feature = my_model["final_test_x"].columns
        model.fit(train_x.loc[:,feature],train_y)
        prediction = np.expm1(model.predict(test_x.loc[:,feature]))
        
        fit_err = mean_squared_error(model.predict(train_x.loc[:,feature]),train_y.values.reshape(-1))
        fit_err = np.sqrt(fit_err)
        result = extract_pd(test_x,prediction)
        result.to_csv("submissions/singlepre_"+key+".csv")
        print(key)
        print(fit_err)
        print(data_help.evaluate(result))
        
        
  