import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


f = open("original data cache/train_x.csv")
train_x = pd.read_csv(f)
train_x.index = train_x.Id
del(train_x["Id"])

f = open("original data cache/test_x.csv")
test_x = pd.read_csv(f)
test_x.index = test_x.Id
del(test_x["Id"])

f = open("original data cache/train_y.csv")
train_y = pd.read_csv(f)
train_y.index = train_y.Id
del(train_y["Id"])

f = open("original data cache/fullscore.csv")
truth = pd.read_csv(f)
truth.index = truth.Id
del(truth["Id"])

def evaluate(sub):
    submission = np.log1p(np.array(sub.SalePrice))
    truth_value = np.log1p(np.array(truth.SalePrice))
    err = mean_squared_error(truth_value,submission)
    
    return np.sqrt(err)
    
    
