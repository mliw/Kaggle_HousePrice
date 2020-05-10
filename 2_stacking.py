import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle
import data_help
N_FOLDS = 10


def extract(file_path):
    with open(file_path,"rb") as f:
        result = pickle.load(f)
    return result   
        
 
def produce_holdout(model,total_data_cache):
    
    train_x = total_data_cache[0].copy()
    train_y = total_data_cache[1].copy()
    train_x = pd.DataFrame(train_x)
    train_y = pd.DataFrame(train_y)
    train_x = train_x.loc[:,model["name"]]
    estimator = model["model"]

    kf = KFold(N_FOLDS, shuffle=True, random_state=42)
    kf.get_n_splits(train_x,train_y)   
    result = pd.DataFrame(np.zeros(train_x.shape[0]), index = train_x.index)
    
    for train_index, test_index in kf.split(train_x, train_y):
        tem_train_x, tem_train_y = train_x.iloc[train_index,:], train_y.iloc[train_index,:]
        tem_test_x, tem_test_y = train_x.iloc[test_index,:], train_y.iloc[test_index,:] 
        tem_train_y = tem_train_y.values.reshape(-1)           
        estimator.fit(tem_train_x,tem_train_y)
        prediction = estimator.predict(tem_test_x)
        result.iloc[test_index,:] = prediction.reshape(-1,1)
        
    return result


def extract_pd(test_x,prediction):    
    result = pd.DataFrame(prediction,index = test_x.index, columns = ["SalePrice"])
    return result


def test_combination(model_list):
    holdout_collect = []
    for model in model_list:
        holdout_collect.append(produce_holdout(model,total_data_cache).values.reshape(-1))
    holdout_collect = np.column_stack(holdout_collect)
    target = train_y.values.reshape(-1)   
    
    # 3.1 Start convex optimization
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
    
    # 3.2 Test holdout prediction
    final_holdout = np.sum(holdout_collect*final_weight,axis = 1)
    holdout_err = np.sqrt(mean_squared_error(train_y,final_holdout))
    
    return final_weight,holdout_err

    
if __name__=="__main__":

    # 1 Load total_data_cache
    train_x = data_help.train_x
    train_y = data_help.train_y
    test_x = data_help.test_x
    total_data_cache = [train_x,train_y,test_x]

    
    # 2 Load single models
    kernel_ridge = "kernel_ridge/17_cv_score_0.12110907975377519.pkl"
    kernel_ridge_1 = "kernel_ridge_1/25_cv_score_0.12388122136649174.pkl"  
    normal_lassocv = "normal_lassocv/21_cv_score_0.11786094415515441.pkl"
    normal_ridge_1 = "normal_ridge_1/22_cv_score_0.1164882792148832_0.11636504910398972.pkl" 
    normal_ridgecv = "normal_ridgecv/21_cv_score_0.1173355741206572_0.11664328989081274.pkl" 
    svr_2 = "svr_2/24_cv_score_0.12511516251872373_0.11768517142989281.pkl" 
    lightgbm = "lightgbm/12_cv_score_0.1285526882743979.pkl"
    xgb = "xgb/14_cv_score_0.12524445314995222.pkl"    
    
    kernel_ridge = extract(kernel_ridge)
    kernel_ridge_1 = extract(kernel_ridge_1)
    normal_lassocv = extract(normal_lassocv)
    normal_ridge_1 = extract(normal_ridge_1)
    normal_ridgecv = extract(normal_ridgecv)
    svr_2 = extract(svr_2)
    lightgbm = extract(lightgbm)
    xgb = extract(xgb)
    
    
    # 3 Test different combination
    model_list = [kernel_ridge,kernel_ridge_1,normal_lassocv,normal_ridge_1,normal_ridgecv,svr_2,lightgbm,xgb]
    key_list = ["kernel_ridge","kernel_ridge_1","normal_lassocv","normal_ridge_1","normal_ridgecv","svr_2","lightgbm","xgb"]
    model_dic = dict(zip(key_list,model_list))
    
    result = {} 
    record_list = []
    while True:
        record_err = float("inf")
        for md_key in key_list:
            tem_record_list = record_list.copy()
            tem_record_list.append(model_dic[md_key])
            weight,err = test_combination(tem_record_list)
            if err<record_err:
                record_err = err
                md_to_add = model_dic[md_key]
                key_to_add = md_key
                print(record_err)
        record_list.append(md_to_add)
        print(len(record_list))
        print("="*30)
        result.update({len(record_list):[record_err,record_list.copy()]})    
        key_list = list(set(key_list)-set([key_to_add]))   
        if len(key_list)==0:
            break
        
    print("="*30)                
    for key in result.keys():      
        print(key,result[key][0])
    print("="*30)
    
    for i in range(1,9):
        final_model_list = result[i][1]
        final_weight,err = test_combination(final_model_list)
        
        # 4 Produce weighted prediction,fitting
        prediction_collect = [] 
        fitting_collect = []    
        for my_model in final_model_list:   
            model = my_model["model"]
            features = my_model["name"]
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
        sub = extract_pd(test_x,final_prediction)
        sub.to_csv("stacking/final_"+str(i)+"_"+str(err)+".csv")
    
    
        
