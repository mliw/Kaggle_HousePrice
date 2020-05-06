import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import data_help
from feature_selection_ga import FeatureSelectionGA
from sklearn.metrics import mean_squared_error
import hyperopt
import pickle
import warnings
warnings.filterwarnings("ignore")
N_FOLDS = 5


# N_FOLDS determines the outcome of this function
def produce_holdout(model,total_data_cache,individual):
    
    train_x = total_data_cache[0].copy()
    train_y = total_data_cache[1].copy()
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)
    individual = np.array(individual).astype(bool)
    train_x = train_x.loc[:,individual]
    
    kf = KFold(N_FOLDS, shuffle=True, random_state=42)
    kf.get_n_splits(train_x,train_y)   
    result = []
    
    for train_index, test_index in kf.split(train_x, train_y):
        tem_train_x, tem_train_y = train_x.iloc[train_index,:], train_y.iloc[train_index,:]
        tem_test_x, tem_test_y = train_x.iloc[test_index,:], train_y.iloc[test_index,:] 
        tem_train_y = tem_train_y.values.reshape(-1)           
        model.fit(tem_train_x,tem_train_y)
        prediction = model.predict(tem_test_x)
        result.append(np.sqrt(mean_squared_error(tem_test_y,prediction)))
    
    return result


def nm_penalty(model,total_data_cache,individual):
        
    train_x = total_data_cache[0].copy()
    train_y = total_data_cache[1].copy()
    
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)
    individual = np.array(individual).astype(bool)
    train_x = train_x.loc[:,individual]   
    
    holdout = produce_holdout(model,total_data_cache,individual)
    result = np.mean(holdout)+np.std(holdout)
    print(result)
    
    return -result


def change_best_ridge(best_p):   
    logi_dic = {0:True,1:False}
    best = best_p.copy()
    best["fit_intercept"] = logi_dic[best["fit_intercept"]]
    best["normalize"] = logi_dic[best["normalize"]]
    best["alphas"] = [best_p["a1"],best_p["a2"],best_p["a3"]]
    del(best["a1"])
    del(best["a2"])
    del(best["a3"])
    
    return best


def extract_pd(test_x,prediction):    
    result = pd.DataFrame(prediction,index = test_x.index, columns = ["SalePrice"])
    return result


if __name__=="__main__":
    
    # 1 Load total_data_cache
    train_x = data_help.train_x
    train_y = data_help.train_y
    test_x = data_help.test_x
    total_data_cache = [train_x,train_y,test_x]
    feature_names = train_x.columns
    

    # 2 Define model and parameters
    normal_ridge = make_pipeline(RobustScaler(), RidgeCV())
    normal_ridge_dic = {
    'a1':hyperopt.hp.uniform('a1',1e-3,20), 
    'a2':hyperopt.hp.uniform('a2',1e-3,20), 
    'a3':hyperopt.hp.uniform('a3',1e-3,20), 
    'fit_intercept':  hyperopt.hp.choice('fit_intercept',[True,False]),   
    'normalize':  hyperopt.hp.choice('normalize',[True,False])
    }

    
    # 3 Genetic Algorithm for feature selection
    key = "normal_ridgecv"
    probability = 0.4
    bench_model = normal_ridge
    length_of_features = train_x.shape[1]
    
    # 4 Start evolving and tunning
    populations = 20
    generations = 5
    
    selector = FeatureSelectionGA(bench_model,total_data_cache,length_of_features,nm_penalty,probability)
    print("70 generations are required to find the best individual. Please wait~~")
    selector.generate(populations,ngen=generations,cxpb=0.1,mutxpb=0.8)

    record = selector.best_generations
    record_score = float("inf")
    result_collect = []
    count = 0
    for i in range(len(record)):
        final_features = record[i]
        final_score = -record[i].fitness.values[0]
        if final_score < record_score:
            record_score = final_score   
            final_features = np.array(final_features).astype(bool)
            final_feature_name = feature_names[final_features]
            
            result = {}
            result.update({"name":final_feature_name})
            result.update({"model":bench_model})
            result.update({"score":record_score}) 
            saved_str = str(count)+"_cv_score_"+str(record_score)
            result.update({"saved_str":saved_str})
            result_collect.append(result)
            count+=1            
            
    # 4.1 Save original models and tunned models
    for items in result_collect:
        saved_str = items["saved_str"]        
        with open(key+"/"+saved_str+".pkl","wb") as f:
            pickle.dump(items,f)
        items["model"].fit(train_x.loc[:,items["name"]],train_y)
        prediction = np.expm1(items["model"].predict(test_x.loc[:,items["name"]]))
        submission = extract_pd(test_x,prediction)  
        submission.to_csv(key+"/"+saved_str+".csv")

        # Tunning model_para
        tunning_train_x = train_x.loc[:,items["name"]] 
        def objective(param):
            fit_intercept = param["fit_intercept"]
            normalize = param["normalize"]
            alphas = np.array([param["a1"],param["a2"],param["a3"]])
            tuning_pipeline = make_pipeline(RobustScaler(),RidgeCV(fit_intercept=fit_intercept,normalize=normalize,alphas=alphas))
            loss = -nm_penalty(tuning_pipeline,[tunning_train_x,train_y],np.ones(tunning_train_x.shape[1]))
            return loss  
        trials = hyperopt.Trials()
        best = hyperopt.fmin(objective,
            space=normal_ridge_dic,
            algo=hyperopt.tpe.suggest,
            max_evals=400,
            trials=trials)      

        best_para = change_best_ridge(best)
        tunned_result = {}
        tunned_result["name"] = items["name"]
        tunned_result["model"] = make_pipeline(RobustScaler(),RidgeCV(**best_para))
        tunned_result["score"] = -nm_penalty(tunned_result["model"],[tunning_train_x,train_y],np.ones(tunning_train_x.shape[1]))
        tunned_result["saved_str"] = saved_str+"_"+str(tunned_result["score"])
        
        with open(key+"/"+tunned_result["saved_str"]+".pkl","wb") as f:
            pickle.dump(tunned_result,f)        
        
        tunned_result["model"].fit(train_x.loc[:,tunned_result["name"]],train_y)
        prediction = np.expm1(tunned_result["model"].predict(test_x.loc[:,tunned_result["name"]]))
        submission = extract_pd(test_x,prediction)  
        submission.to_csv(key+"/"+tunned_result["saved_str"]+".csv")
    
       
        