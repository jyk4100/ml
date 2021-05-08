## python script for xgboost toy example to compare against R
## on work setup windows desktop R 4.0 xgboost 1.3.1 and unix server py 3.8.3 and xgboost 1.2.1 
## predictions were different with same hyperparams 
## happens to be so when sampling parameters are set to less than 0 (i.e. sampling)
## check same env and pkg levels
## https://github.com/dmlc/xgboost/issues/6941
## jyk4100
## last modified: 2021-05-04

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
xgb.__version__ ## 1.4.1

## set path and read data
os.chdir('''C:/Users/Kim Jungyoon/Documents/2.study/ml/xgboost/''')
os.listdir()
# iris = pd.read_csv('''iris.csv''')

## or read from url
iris = pd.read_csv("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
## toy example
iris['class'] = iris['species'].apply(lambda x: 1 if x == 'setosa' else 0)
pd.crosstab(iris['class'], iris['species'])

## define and train model
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
param = {'objective': 'binary:logistic', 'tree_method':'approx','scale_pos_weight':1,
         'n_estimators':10, 'eta':0.3, 'gamma':5, 'random_state':123, 
         'max_depth':6, 'min_child_weight':1, 
         'reg_lambda':0.5, 'reg_alpha':0.5,
         'sampling_method':'uniform', 'colsample_bytree':1.0, 'subsample':1.0}
model = XGBClassifier(**param)
model.fit(iris[features].values, iris['class'])
model.get_params()

## predicted prob
preds = pd.Series(model.predict_proba(iris[features].values)[:,1]).round(5)
preds.astype(str).value_counts()


## R vs py same results until these params
param = {'objective': 'binary:logistic', 'tree_method':'approx','scale_pos_weight':1,
         'n_estimators':10, 'eta':0.3, 'gamma':5, 'random_state':123, 
         'max_depth':6, 'min_child_weight':1, 
         'reg_lambda':0.5, 'reg_alpha':0.5}
## "same params"
# 0.03662    100
# 0.95171     50

## "same params" + colsample_bytree:1.0, subsample:1.0 (same anyways since default 1.0)
# 0.03662    100
# 0.95171     50

## "same params" + colsample_bytree:0.7, subsample:1.0
# 0.0369     93
# 0.95078    45
# 0.06652     7
# 0.91216     5
## "same params" + colsample_bytree:1.0, subsample:0.8
# 0.0448     99
# 0.93084    50
# 0.07157     1
## could wrap in function but no need detrimental for "bug" reporting purpose