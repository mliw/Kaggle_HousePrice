import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
N_FOLDS = 10


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


def test_feature(model,train_x,train_y,):
    
    holdout = produce_holdout(model,train_x,train_y)
    err = mean_squared_error(train_y,holdout)
    
    return np.sqrt(err)


def detect_feature(model,train_data,feature):
    
    tem_data = train_data.loc[:,[feature,"SalePrice"]]
    tem_data.dropna(axis=0,inplace=True)
    train_x = tem_data[feature]
    train_y = tem_data["SalePrice"]
    
    # 1 Start test feature power
    if str(train_x.dtype)=="object":
        print("object feature!")
        train_x_0 = pd.get_dummies(train_x)
        power = test_feature(model,train_x_0,train_y)
        print("(dummy)Feature power is {}".format(power))
        
        lb = LabelEncoder()
        train_x_1 = lb.fit_transform(train_x)
        power = test_feature(model,train_x_1,train_y)
        print("(LabelEncoder)Feature power is {}".format(power))
        train_x_1 = pd.DataFrame(train_x_1,index = train_x.index)
        pic = pd.concat([train_x_1,train_y],axis = 1)  
    else:
        print("Numerical feature!")
        power = test_feature(model,train_x,train_y)
        print("Feature power is {}".format(power))
        pic = pd.concat([train_x,train_y],axis = 1)   
            
    # 2 Calculate missing values
    missing = train_data[feature].isnull().sum()
    print("Missing value is {}".format(missing))
    
    pic.columns = [feature,"SalePrice"]
    sns.set()
    sns.relplot(x=feature,y="SalePrice",data=pic)
    plt.show()
    
    return missing,power

    
def test_feature_v1(train_test,train_data,train_id,feature,model):
    tem_train = train_test.loc[train_id,:]
    tem_train["SalePrice"] = train_data.loc[train_id,"SalePrice"]       
    
    return detect_feature(model,tem_train,feature)
    

if __name__=="__main__":
    
    # 0 Load the data
    f = open("original data cache/train.csv")
    train_data = pd.read_csv(f)
    train_data.SalePrice = np.log1p(train_data.SalePrice)
    f = open("original data cache/test.csv")
    test_data = pd.read_csv(f)
    train_data.index = train_data['Id']
    test_data.index = test_data['Id']
    train_id = train_data['Id']
    test_id = test_data['Id']
    train_y = np.log1p(train_data.SalePrice)
    del(train_data['Id'])
    del(test_data['Id'])
    all_data = pd.concat((train_data, test_data))
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    print("all_data size is : {}".format(all_data.shape))
    
    
    # 1 Feature engineering
    model = XGBRegressor(n_jobs=3)
    train_test = all_data.copy()
    
    # 1.1 White feature
    train_data["bench"]=1
    feature = 'bench'
    detect_feature(model,train_data,feature)
    del(train_data["bench"])
    
    # 1.2 Impute missing feature
    missing_feature = []
    except_feature = []
    feature_score = []
    for feature in all_data.columns:
        print("="*30)
        try:
            print(feature)
            miss,score = detect_feature(model,train_data,feature)
            if miss>0:
                missing_feature.append(feature)
            feature_score.append([feature,score])
        except:
            except_feature.append(feature)
        print("="*30) 
    feature_score = pd.DataFrame(feature_score,columns=["name","score"])
    
    """
    We have 18 missing features at this time
    ['LotFrontage','Alley','MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
 'Electrical','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','Fence','MiscFeature']
    """
    feature = 'LotFrontage'
    detect_feature(model,train_data,feature)    
    train_test["LotFrontage"] = train_test.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())) 
    
    train_test["Alley"] = train_test["Alley"].fillna("None")   
    train_test["MasVnrType"] = train_test["MasVnrType"].fillna("None")
    train_test["MasVnrArea"] = train_test["MasVnrArea"].fillna(train_test["MasVnrArea"].median())    
    train_test["BsmtQual"] = train_test["BsmtQual"].fillna("no")     
    train_test["BsmtCond"] = train_test["BsmtCond"].fillna("no")     
    train_test["BsmtExposure"] = train_test["BsmtExposure"].fillna("nobase")     
    train_test["BsmtFinType1"] = train_test["BsmtFinType1"].fillna("nobase") 
    train_test["BsmtFinType2"] = train_test["BsmtFinType2"].fillna("nobase") 
    train_test["Electrical"] = train_test["Electrical"].fillna(train_test["Electrical"].mode()[0]) 
    train_test["FireplaceQu"] = train_test["FireplaceQu"].fillna(train_test["FireplaceQu"].mode()[0]) 
    train_test["GarageType"] = train_test["GarageType"].fillna("nogarage") 
    train_test["GarageYrBlt"] = train_test["GarageYrBlt"].fillna(train_test["GarageYrBlt"].median()) 
    train_test["GarageFinish"] = train_test["GarageFinish"].fillna("nogarage") 
    train_test["GarageQual"] = train_test["GarageQual"].fillna("nogarage") 
    train_test["GarageCond"] = train_test["GarageCond"].fillna("nogarage") 
    train_test["Fence"] = train_test["Fence"].fillna("nofence") 
    train_test["MiscFeature"] = train_test["MiscFeature"].fillna("None") 

    remaining_missing = list(train_test.isnull().sum().index[train_test.isnull().sum()>0])
    """
    We have 16 missing features at this time
['MSZoning','Utilities','Exterior1st','Exterior2nd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF',
 'BsmtFullBath','BsmtHalfBath','KitchenQual','Functional','GarageCars','GarageArea','PoolQC','SaleType']
    """
    train_test["MSZoning"] = train_test["MSZoning"].fillna(train_test["MSZoning"].mode()[0])
    train_test["Utilities"] = train_test["Utilities"].fillna(train_test["Utilities"].mode()[0])
    train_test["Exterior1st"] = train_test["Exterior1st"].fillna(train_test["Exterior1st"].mode()[0])
    train_test["Exterior2nd"] = train_test["Exterior2nd"].fillna(train_test["Exterior2nd"].mode()[0])
    train_test["BsmtFinSF1"] = train_test["BsmtFinSF1"].fillna(train_test["BsmtFinSF1"].median())
    train_test["BsmtFinSF2"] = train_test["BsmtFinSF2"].fillna(train_test["BsmtFinSF2"].median())
    train_test["BsmtUnfSF"] = train_test["BsmtUnfSF"].fillna(train_test["BsmtUnfSF"].median())
    train_test["TotalBsmtSF"] = train_test["TotalBsmtSF"].fillna(train_test["TotalBsmtSF"].median())
    train_test["BsmtFullBath"] = train_test["BsmtFullBath"].fillna(train_test["BsmtFullBath"].mode()[0])
    train_test["BsmtHalfBath"] = train_test["BsmtHalfBath"].fillna(train_test["BsmtHalfBath"].mode()[0])
    train_test["KitchenQual"] = train_test["KitchenQual"].fillna(train_test["KitchenQual"].mode()[0])
    train_test["Functional"] = train_test["Functional"].fillna(train_test["Functional"].mode()[0])
    train_test["GarageCars"] = train_test["GarageCars"].fillna(train_test["GarageCars"].median())
    train_test["GarageArea"] = train_test["GarageArea"].fillna(0)
    train_test["PoolQC"] = train_test["PoolQC"].fillna("nopool")
    train_test["SaleType"] = train_test["SaleType"].fillna(train_test["SaleType"].mode()[0])
    print("The number of nan is {}".format(train_test.isnull().sum().sum()))

    # 1.3 Test the power of each single feature
    train_x = train_test.loc[train_id,:]
    train_x["SalePrice"] = train_data.loc[train_id,"SalePrice"]
    missing_feature = []
    except_feature = []
    feature_score = []
    for feature in all_data.columns:
        print("="*30)
        try:
            print(feature)
            miss,score = detect_feature(model,train_x,feature)
            if miss>0:
                missing_feature.append(feature)
            feature_score.append([feature,score])
        except:
            except_feature.append(feature)
        print("="*30) 
    feature_score = pd.DataFrame(feature_score,columns=["name","score"])    
    
    # 1.4 Design new features based on those excellent features
    small_index = feature_score["score"].nsmallest(20).index
    print(list(feature_score.loc[small_index,"name"]))
    print("="*30)
    print("Divide features into different categories:")
    print("""
    {'OverallQual':Rates the overall material and finish of the house}
    {'Neighborhood':Physical locations within Ames city limits}
    {'GarageCars':"Size of garage in car capacity",'GarageArea': "Size of garage in square feet",'GarageFinish': "Interior finish of the garage" \
        'GarageYrBlt': "Year garage was built", 'GarageType': "Garage location"}
    {'ExterQual':Evaluates the quality of the material on the exterior,'BsmtQual': Evaluates the height of the basement \
        'Foundation': "Type of foundation"}
    {'KitchenQual': Kitchen quality}
    {'GrLivArea': Above grade (ground) living area square feet,'TotRmsAbvGrd':"Total rooms above grade (does not include bathrooms)", \
        'FullBath': "Full bathrooms above grade",'1stFlrSF': "First Floor square feet",'TotalBsmtSF: "Total square feet of basement area"}
    {'YearBuilt': "Original construction date",'YearRemodAdd': "Remodel date (same as construction date if no remodeling or additions)"}
    {'MSSubClass': "Identifies the type of dwelling involved in the sale."}
    {'Fireplaces': "Number of fireplaces"}
    """)
    print("="*30)

    # 1.4.1 Explore garage
    garage_words = ['GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond']   
    tem_garage = train_test[garage_words]
    def area_per_car(clause):
        result = clause['GarageArea'] / clause['GarageCars'] if clause['GarageCars']!=0 else 0
        return result
    train_test["area_per_car"] = train_test.apply(area_per_car,axis = 1)

    # 1.4.2 Explore area
    train_test["above_and_ground_area"] = train_test["TotalBsmtSF"]+train_test["GrLivArea"]
    train_test['Total_Bathrooms'] = (train_test['FullBath'] + (0.5 * train_test['HalfBath']) +
                               train_test['BsmtFullBath'] + (0.5 * train_test['BsmtHalfBath']))
    train_test["one_and_two"] = train_test["1stFlrSF"]+train_test["2ndFlrSF"]
    train_test['Total_Porch_Area'] = (train_test['OpenPorchSF'] + train_test['3SsnPorch'] + train_test['EnclosedPorch'] + train_test['ScreenPorch'] + train_test['WoodDeckSF'])

    # 1.5 Delete outliers of important features
    train_x = train_test.loc[train_id,:]
    train_x["SalePrice"] = train_data.loc[train_id,"SalePrice"]
    missing_feature = []
    except_feature = []
    feature_score = []
    feature_to_test = list(set(train_x.columns)-set(["SalePrice"]))
    for feature in feature_to_test:
        print("="*30)
        try:
            print(feature)
            miss,score = detect_feature(model,train_x,feature)
            if miss>0:
                missing_feature.append(feature)
            feature_score.append([feature,score])
        except:
            except_feature.append(feature)
        print("="*30) 
    feature_score = pd.DataFrame(feature_score,columns=["name","score"])     
    small_index = feature_score["score"].nsmallest(20).index
    print(list(feature_score.loc[small_index,"name"]))        
    """
    t1 = ['OverallQual', 'above_and_ground_area', 'Neighborhood', 'GarageCars', 'Total_Bathrooms', \
        'ExterQual', 'BsmtQual', 'one_and_two', 'KitchenQual', 'GrLivArea', 'GarageArea', \
        'YearBuilt', 'GarageFinish', 'GarageYrBlt', 'FullBath', 'TotalBsmtSF', 'GarageType', \
            'MSSubClass', 'YearRemodAdd', 'Foundation']
    """
    train_test_save = train_test.copy()
    tem_train = train_test.loc[train_id,:]
    tem_train["SalePrice"] = train_data.loc[train_id,"SalePrice"]
     
    logi_0 = np.logical_and(tem_train.SalePrice>=12.25, tem_train.OverallQual==4)
    logi_1 = np.logical_and(tem_train.SalePrice<=11.5, tem_train.OverallQual==7)    
    logi_2 = np.logical_and(tem_train.SalePrice<=12.5, tem_train.OverallQual==10)
    logi_3 = np.logical_and(tem_train.SalePrice<=12.5, tem_train.above_and_ground_area>=6000)   
    logi_4 = np.logical_and(tem_train.SalePrice<=11, tem_train.ExterQual=="Gd")     
    logi_5 = np.logical_and(tem_train.SalePrice<=12.5, tem_train.one_and_two>=4000)    
    logi_6 = np.logical_and(tem_train.SalePrice<=11.5, tem_train.GarageArea>=1200)       
    logi_7 = np.logical_and(tem_train.SalePrice<=12.5, tem_train.TotalBsmtSF>=5000)    
    logi = logi_0
    
    for i in range(8):
        exec("""logi = np.logical_or(logi,logi_"""+str(i)+""")""")

    tem_train.drop(index = tem_train.index[logi],axis = 0,inplace=True)
    tem_test = train_test.loc[test_id,:]
    train_id = tem_train.index
    test_id = tem_test.index
    
    train_x = tem_train.loc[:,tem_test.columns]
    train_y = pd.DataFrame(tem_train["SalePrice"])
    test_x = tem_test
    
    train_test_x = pd.concat([train_x,test_x],axis = 0)
    train_test_x = pd.get_dummies(train_test_x)
    
    train_x = train_test_x.loc[train_id,:]
    test_x = train_test_x.loc[test_id,:]
    
    train_x.to_csv("original data cache/train_x.csv")
    test_x.to_csv("original data cache/test_x.csv")
    train_y.to_csv("original data cache/train_y.csv")

