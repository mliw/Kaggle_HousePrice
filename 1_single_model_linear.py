import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
import data_help
from feature_selection_ga import FeatureSelectionGA
from model_help import model_opt
from sklearn.metrics import mean_squared_error
import hyperopt
import pickle
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
    linear_model = make_pipeline(RobustScaler(), LinearRegression())

    
    # 3 Genetic Algorithm for feature selection
    key = "linear"
    probability = 0.4
    bench_model = linear_model
    length_of_features = train_x.shape[1]
    
    #for rd in range(3):
    for rd in [0,1]:
        selector = FeatureSelectionGA(bench_model,total_data_cache,length_of_features,nm_penalty,probability)
        print("40 generations are required to find the best individual. Please wait~~")
        selector.generate(600,ngen=70,cxpb=0.1,mutxpb=0.8)
    
        record = selector.best_generations
        record_score = float("inf")
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
                with open(key+"/round_"+str(rd)+"_item_"+str(count)+"_cv_score_"+str(record_score)+".pkl","wb") as f:
                    pickle.dump(result,f)
                    
                result["model"].fit(train_x.loc[:,final_feature_name],train_y)
                prediction = np.expm1(result["model"].predict(test_x.loc[:,final_feature_name]))
                submission = extract_pd(test_x,prediction)  
                submission.to_csv(key+"/round_"+str(rd)+"_item_"+str(count)+"_cv_score_"+str(record_score)+".csv")
                count+=1
                print("="*60)
                print("cv_score is:")
                print(record_score)
                print("="*60)
           
  