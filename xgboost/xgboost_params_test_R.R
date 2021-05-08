## R script for xgboost toy example to compare against Python
## on work setup windows desktop R 4.0 xgboost 1.3.1 and unix server py 3.8.3 and xgboost 1.2.1 
## predictions were different with same hyperparams 
## happens to be so when sampling parameters are set to less than 0 (i.e. sampling)
## check same env and pkg levels
## https://github.com/dmlc/xgboost/issues/6941
## jyk4100
## last modified 2021-05-05
## turns out R own rng vs C rng

library(data.table)
library(xgboost)
sessionInfo() ## 1.4.1.1

## load data
# data_path = model_path = "C:/Users/Kim Jungyoon/Documents/2.study/ml/xgboost/"
# iris = fread(paste0(data_path, "iris.csv"))
## read data from url
iris = fread("https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv")
## create dummy class
iris[, c("class") := ifelse(iris$species == "setosa", 1, 0)]

## define and training model
set.seed(123)
model = xgboost(data=data.matrix(iris[, c("sepal_length", "sepal_width", "petal_length", "petal_width")]),
                label=iris$class, verbose=0, 
                objective="binary:logistic", tree_method="approx", scale_pos_weight=1,
                nround=10, eta=0.3, gamma=5,
                max_depth=6, min_child_weight=1,
                lambda=0.5, alpha=0.5, 
                sampling_method="uniform", colsample_bytree=1.0, subsample=1.0)
## prediction on irising set
preds = round(predict(model, data.matrix(iris[, c("sepal_length", "sepal_width", "petal_length", "petal_width")])), 5)
preds[c(1:10,140:150)]
table(preds)

## "same params"
# 0.03662 0.95171 
# 100      50 
## "same params" + colsample_bytree:0.7, subsample:1.0
# 0.04464 0.07019 0.09411 0.15894 0.85324 0.88889 0.93571 
# 90       3       6       1       5       1      44 
## "same params" + colsample_bytree:1.0, subsample:0.8
# 0.04696 0.93423 
# 100      50 

## could wrap in function but no need detrimental for "bug" reporting purpose