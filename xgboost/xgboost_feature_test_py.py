## python script for xgboost feature list vs set test toy example
## on work setup windows desktop python 3.8.3 & xgboost 1.2.1
## debugged inconsistent model results from model training job
## found out someone put {} instead of [] while passing feature list
## ohhh set orders since its set so data was basically shuffled...
## nothing to do with xgboost lol
## jyk4100
## last modified: 2021-05-07

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
xgb.__version__ ## 1.4.1

## set path and read data
os.chdir('''C:/Users/Kim Jungyoon/Documents/2.study/ml/xgboost/''')
os.listdir()
iris = pd.read_csv('''iris.csv''')

## toy example
iris['class'] = iris['species'].apply(lambda x: 1 if x == 'setosa' else 0)
pd.crosstab(iris['class'], iris['species'])

## define and train model
param = {'objective': 'binary:logistic', 'tree_method':'approx','scale_pos_weight':1,
         'n_estimators':10, 'eta':0.3, 'gamma':5, 'random_state':123, 
         'max_depth':6, 'min_child_weight':1, 
         'reg_lambda':0.5, 'reg_alpha':0.5,
         'sampling_method':'uniform', 'colsample_bytree':1.0, 'subsample':1.0}

## 1. "passing" features as list
features1 = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
model1 = XGBClassifier(**param).fit(iris[features1].values, iris['class'])
## predicted probs
preds1 = pd.Series(model1.predict_proba(iris[features1].values)[:,1]).round(5)
preds1.value_counts()
preds1_2 = pd.Series(model1.predict_proba(iris[features2].values)[:,1]).round(5)
preds1_2.value_counts()

## 2. features as dict?
features2 = {"sepal_length", "sepal_width", "petal_length", "petal_width"}
print("list subset df object type:{}", type(iris[features1]))
print("set subset df object type:{}", type(iris[features2]))
model2 = XGBClassifier(**param).fit(iris[features2].values, iris['class'])
## predicted probs
preds2 = pd.Series(model2.predict_proba(iris[features2].values)[:,1]).round(5)
preds2.value_counts()
preds2_1 = pd.Series(model2.predict_proba(iris[features1].values)[:,1]).round(5)
preds2_1.value_counts()

## compare
(preds2 == preds1).value_counts()
## same okay old version issue
