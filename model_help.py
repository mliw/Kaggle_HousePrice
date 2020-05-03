# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 21:06:43 2019

@author: mingwei
"""

# 0 Import packages
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
import hyperopt
from hyperopt import fmin, Trials
from tqdm import tqdm
import pickle
N_FOLDS = 10


# N_FOLDS determines the outcome of this function
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
    
    return result


def nm_cv(model,train_x,train_y):
    
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)

    holdout = produce_holdout(model,train_x,train_y)
    truth = train_y
    err = np.sqrt(mean_squared_error(truth,holdout))
    
    model.fit(train_x,train_y)
    fitting = np.sqrt(mean_squared_error(truth,model.predict(train_x))) 
    
    penalty = err-fitting
    result = err+penalty if penalty>0 else err
    return result


class average_model:
    
    def __init__(self,models,train_test_batches,current_features,train_y,current_prediction=None):
        
        self.models = models
        self.train_test_batches = train_test_batches
        self.current_features = current_features
        self.current_prediction = current_prediction
        
        self.best_score = None
        self.holdout_fit = None
        self.best_holdout_prediction = None

        holdout_index = train_y.index
        self.holdout_prediction = pd.DataFrame(np.zeros((len(holdout_index),len(models))),index = holdout_index)
        self.holdout_truth = train_y        
        
        
    def get_holdout_prediction(self):
        
        for i in range(len(self.models)):
            print("Model {} is fitting.".format(i))
            for tem_batch in self.train_test_batches:
                tem_train_x = tem_batch[0].loc[:,self.current_features]
                tem_train_y = tem_batch[1]
                tem_test_x = tem_batch[2].loc[:,self.current_features]
                self.models[i].fit(tem_train_x,tem_train_y.values.reshape(-1))
                tem_prediction = self.models[i].predict(tem_test_x)
                self.holdout_prediction.loc[tem_batch[2].index,i] = tem_prediction
            
    
    def get_holdout_fit(self): 
        
        self.get_best_weight()
        self.holdout_fit = [items.copy() for items in self.train_test_batches]            
        print("Prepare data for next iteration!")        
        for tem_batch in self.holdout_fit:
            tem_train_x = tem_batch[0].loc[:,self.current_features]
            tem_train_y = tem_batch[1]
            tem_test_x = tem_batch[2].loc[:,self.current_features]
            for i in range(len(self.models)):
                print("Model {} is fitting.".format(i))
                self.models[i].fit(tem_train_x,tem_train_y.values.reshape(-1))
            
            fit_array = np.column_stack([self.models[i].predict(tem_train_x) for i in range(len(self.models))])
            fit_array = np.sum(fit_array * self.best_weight, axis = 1)
            tem_fit = pd.DataFrame(fit_array,index = tem_train_y.index)
            tem_residual = pd.DataFrame(tem_train_y.values.reshape(-1)-tem_fit.values.reshape(-1),index = tem_train_y.index)
            tem_batch[1] = tem_residual
            
                
    def get_best_weight(self):
        
        self.get_holdout_prediction()
        weight_dic = {}
        for i in range(len(self.models)):
            weight_dic.update({"w"+str(i): hyperopt.hp.uniform("w"+str(i),1e-4,1)})
        
        def objective(param):
            my_weight = []
            for i in range(len(param.keys())):
                my_weight.append(param["w"+str(i)])
            my_weight = np.array(my_weight) / np.sum(my_weight)
            combined_prediction = np.sum(self.holdout_prediction.values*my_weight,axis = 1)
            
            if self.current_prediction is not None:
                self.current_prediction = pd.DataFrame(self.current_prediction)
                tem_pre = self.current_prediction.loc[self.holdout_prediction.index,:].values.reshape(-1)
                combined_prediction+=tem_pre
            
            err = mean_squared_error(self.holdout_truth.values.reshape(-1),combined_prediction)
            return np.sqrt(err)

        trials = Trials()
        print("Start optimizing weights!")
        best = fmin(objective,
            space=weight_dic.copy(),
            algo=hyperopt.tpe.suggest,
            max_evals=4000,
            trials=trials)
        
        final_weight = []
        for (_,value) in best.items():
            final_weight.append(value)
        final_weight = np.array(final_weight) / np.sum(final_weight)
        
        self.best_weight = final_weight
        self.best_score = trials.best_trial["result"]["loss"]
        self.best_holdout_prediction = np.sum(self.holdout_prediction*self.best_weight,axis = 1)
        self.best_holdout_prediction = pd.DataFrame(self.best_holdout_prediction)
        if self.current_prediction is not None:
            self.current_prediction = pd.DataFrame(self.current_prediction)
            tem_pre = self.best_holdout_prediction.loc[self.holdout_prediction.index,:].values.reshape(-1)
            combined_prediction=tem_pre+self.current_prediction.values.reshape(-1)
            self.best_holdout_prediction = pd.DataFrame(combined_prediction,index = self.holdout_prediction.index)
                    
        
def select_features(bench_model,train_test_batches,train_y,total_features,previous_prediction=None):
    current_combination = []
    result = {}
    while True:
        features_to_add = set(total_features)-set(current_combination)
        features_to_add = list(features_to_add)
        if len(features_to_add)==0:
            break
        best_score = float("inf")
        best_tem_features = None
        try:
            with tqdm(features_to_add) as total:
                for feature in total:
                    tem_features = current_combination.copy()
                    tem_features.append(feature)
                    tem_score = bt_cv(bench_model,train_test_batches,train_y,tem_features,previous_prediction) 
                    if tem_score<best_score:
                        print("Current best score is {}".format(tem_score))
                        best_score = tem_score
                        best_tem_features = tem_features.copy()
                current_combination = best_tem_features.copy()
                result.update({len(current_combination):[best_score,best_tem_features]})
        except:
            pass
        num = len(current_combination)
        if num>1 and result[num][0]>result[num-1][0]-0.0001:
            break

    current_result = result[num-1] if num>=1 else result[num]
    print("Best score of this iteration is {}".format(current_result[0]))
        
    return current_result

  
class model_opt:
    
    def __init__(self,model,feature_names,train_x,train_y,test_x,model_para_dic,num_of_trials):   
        self.model = model
        self.feature_names = feature_names
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.model_para_dic = model_para_dic
        self.num_of_trials = num_of_trials 
        
        self.best_para = None  
        self.best_score = None
        self.final_para = None
        self.final_model = None 
        self.final_score = None
        self.loaded_package = None
        
    def find_best_para(self):
        
        train_x = self.train_x.loc[:,self.feature_names] 
        train_x = train_x.values
        train_y = self.train_y
        
        def objective(param):
            tuning_pipeline = self.model(**param)
            loss = nm_cv(tuning_pipeline,train_x,train_y)
            return loss  

        trials = Trials()
        best = fmin(objective,
            space=self.model_para_dic,
            algo=hyperopt.tpe.suggest,
            max_evals=self.num_of_trials,
            trials=trials)
        
        self.best_para = best
        
    def find_best_para_svr(self):
        
        train_x = self.train_x.loc[:,self.feature_names] 
        train_x = train_x.values
        train_y = self.train_y
        
        def objective(param):
            tuning_pipeline = make_pipeline(RobustScaler(),self.model(**param))
            loss = nm_cv(tuning_pipeline,train_x,train_y)
            return loss  

        trials = Trials()
        best = fmin(objective,
            space=self.model_para_dic,
            algo=hyperopt.tpe.suggest,
            max_evals=self.num_of_trials,
            trials=trials)
        
        self.best_para = best
        
    
    def set_final_para(self,param):
        self.final_para = param
        self.final_model = self.model(**param)
        train_x = self.train_x.loc[:,self.feature_names] 
        train_x = train_x.values
        train_y = self.train_y  
        self.final_score = nm_cv(self.final_model,train_x,train_y)
        
    def set_final_para_svr(self,param):
        self.final_para = param
        self.final_model = make_pipeline(RobustScaler(),self.model(**param))
        train_x = self.train_x.loc[:,self.feature_names] 
        train_x = train_x.values
        train_y = self.train_y  
        self.final_score = nm_cv(self.final_model,train_x,train_y)
        
    
    def save_model(self,path):
        self.loaded_package = {"final_score":self.final_score,"final_model":self.final_model,"final_para":self.final_para, \
                      "final_train_x":self.train_x.loc[:,self.feature_names],"final_test_x":self.test_x.loc[:,self.feature_names]}
        
        self.final_model.fit(self.loaded_package["final_train_x"],self.train_y)
        
        self.loaded_package.update({"final_prediction":self.final_model.predict(self.test_x.loc[:,self.feature_names])})
        
        if self.final_score is None or self.final_model is None or self.final_para is None:
            print("Please execute set_final_para to set final paramter!")
            
        with open(path,"wb") as f:
            pickle.dump(self.loaded_package,f)   
    
    def load_model(self,path):
        with open(path,"rb") as f:
            self.loaded_package = pickle.load(f)
            
    def output_result(self,path):
        
        if self.loaded_package == None:
            print("Please save_model first! loaded_package is None!")
            return
        
        tem = self.loaded_package
        result = pd.DataFrame({"SalePrice":np.expm1(tem["final_prediction"])},index = tem["final_test_x"].index)
   
        result.to_csv(path)     

           
         
                    