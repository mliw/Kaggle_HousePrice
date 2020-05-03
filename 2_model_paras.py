# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:06:43 2019

@author: mingwei
"""

# 0 Import packages
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge,LassoCV
from lightgbm import LGBMRegressor
import data_help
import pickle
import hyperopt
from model_help import model_opt


def change_best_XGBRegressor(best_p):
    best = best_p.copy()
    best["max_depth"]+=1
    best["min_child_weight"]+=1    
    best["n_estimators"]+=1  
    best["n_jobs"]=3
    best["objective"]="reg:squarederror"

    return best


def change_best_ridge(best_p):   
    logi_dic = {0:True,1:False}
    best = best_p.copy()
    best["fit_intercept"] = logi_dic[best["fit_intercept"]]
    best["normalize"] = logi_dic[best["normalize"]]
    
    return best

def change_best_lasso(best_p):   
    best_list = [best_p[keys] for keys in best_p.keys()]
    best_dic = {"alphas":best_list}
    return best_dic

def change_best_lgbm(best_p):
    best = best_p.copy()
    best["max_depth"]+=1  
    best["n_estimators"]+=1  
    best["n_jobs"]=3

    return best


if __name__=='__main__':

    # 1 Load data
    train_x = data_help.train_x
    test_x = data_help.test_x
    train_y = data_help.train_y
    
    
    # 2 Start optimizing xgb
    with open("models/model_xgb.pkl","rb") as f:
        my_model = pickle.load(f)
    
    # 2.1 Define model_para_dic   
    XGBRegressor_dic = {
        'learning_rate': hyperopt.hp.uniform('learning_rate',0,2),
        'reg_alpha': hyperopt.hp.uniform('reg_alpha',0,2),
        'reg_lambda': hyperopt.hp.uniform('reg_lambda',0,2),
        'gamma': hyperopt.hp.uniform('gamma',0,0.5),
        'n_jobs': hyperopt.hp.choice('n_jobs',[3]),
        'max_depth': hyperopt.hp.randint('max_depth',7)+1,
        'min_child_weight': hyperopt.hp.randint('min_child_weight',7)+1,  
        'n_estimators': hyperopt.hp.randint('n_estimators',150)+1,    
        'subsample': hyperopt.hp.uniform('subsample',0,1),
        'objective': hyperopt.hp.choice('objective',["reg:squarederror"]),
    }
            
    # 2.2 Optimization 
    standard_features = my_model["name"]
    optimization_object = model_opt(XGBRegressor,standard_features,train_x,train_y,test_x,XGBRegressor_dic,3000)
    optimization_object.find_best_para()
    tem_best = optimization_object.best_para
    optimization_object.set_final_para(change_best_XGBRegressor(tem_best))
    optimization_object.save_model("tunned_models/model_xgb.pkl")


    # 3 Start optimizing svr
    with open("models/model_svr.pkl","rb") as f:
        my_model = pickle.load(f)
    
    # 3.1 Define model_para_dic   
    svr_dic = {
        'gamma':hyperopt.hp.uniform('gamma',1e-5,1e-2), 
        'C':  hyperopt.hp.uniform('C',1e-2,40),   
        'epsilon':  hyperopt.hp.uniform('epsilon',1e-4,1e-2)
    }
    
    # 3.2 Optimization 
    standard_features = my_model["name"]
    optimization_object = model_opt(SVR,standard_features,train_x,train_y,test_x,svr_dic,3000)
    optimization_object.find_best_para_svr()
    tem_best = optimization_object.best_para
    optimization_object.set_final_para_svr(tem_best)
    optimization_object.save_model("tunned_models/model_svr.pkl")


    # 4 Start optimizing ridge
    with open("models/model_normal_ridge.pkl","rb") as f:
        my_model = pickle.load(f)
    
    # 4.1 Define model_para_dic   
    ridge_dic = {
        'alpha':hyperopt.hp.uniform('alpha',0,7), 
        'fit_intercept':  hyperopt.hp.choice('fit_intercept',[True,False]),   
        'normalize':  hyperopt.hp.choice('normalize',[True,False])    
    }
    
    # 4.2 Optimization 
    standard_features = my_model["name"]
    optimization_object = model_opt(Ridge,standard_features,train_x,train_y,test_x,ridge_dic,1000)
    optimization_object.find_best_para_svr()
    tem_best = optimization_object.best_para
    optimization_object.set_final_para_svr(change_best_ridge(tem_best))
    optimization_object.save_model("tunned_models/model_normal_ridge.pkl")
    

    # 5 Start optimizing model_lightgbm
    with open("models/model_lightgbm.pkl","rb") as f:
        my_model = pickle.load(f)
        
    
    # 5.1 Define model_para_dic   
    LGBMRegressor_dic = {
        'learning_rate': hyperopt.hp.uniform('learning_rate',0,2),
        'reg_alpha': hyperopt.hp.uniform('reg_alpha',0,2),
        'reg_lambda': hyperopt.hp.uniform('reg_lambda',0,2),
        'n_jobs': hyperopt.hp.choice('n_jobs',[3]),
        'max_depth': hyperopt.hp.randint('max_depth',11)+1, 
        'n_estimators': hyperopt.hp.randint('n_estimators',250)+1,    
        'subsample': hyperopt.hp.uniform('subsample',0,1)
    }
    
    # 5.2 Optimization 
    standard_features = my_model["name"]
    optimization_object = model_opt(LGBMRegressor,standard_features,train_x,train_y,test_x,LGBMRegressor_dic,3000)
    optimization_object.find_best_para()
    tem_best = optimization_object.best_para
    optimization_object.set_final_para(change_best_lgbm(tem_best))
    optimization_object.save_model("tunned_models/model_lightgbm.pkl")
    
    
    # 6 Start optimizing model_lasso
    with open("models/model_lasso.pkl","rb") as f:
        my_model = pickle.load(f)
  
    
    # 6.1 Define model_para_dic   
    lasso_dic = {
        'alpha0':hyperopt.hp.uniform('alpha0',0,1), 
        'alpha1':hyperopt.hp.uniform('alpha1',0,1),   
        'alpha2':hyperopt.hp.uniform('alpha2',0,1),
        'alpha3':hyperopt.hp.uniform('alpha3',0,1),   
        'alpha4':hyperopt.hp.uniform('alpha4',0,1),
        'alpha5':hyperopt.hp.uniform('alpha5',0,1), 
    }
    
    # 6.2 Optimization 
    standard_features = my_model["name"]
    optimization_object = model_opt(LassoCV,standard_features,train_x,train_y,test_x,lasso_dic,1000)
    optimization_object.find_best_para_lasso()
    tem_best = optimization_object.best_para
    optimization_object.set_final_para(change_best_lasso(tem_best))
    optimization_object.save_model("tunned_models/model_lasso.pkl")   
    

    
    
    
    
    
    
    