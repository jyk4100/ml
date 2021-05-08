## R script for xgboost toy example to compare with xgboost4j-spark
## jyk4100
## last modified 2020-06-25

library(data.table)
library(xgboost)

## data.table::fread <3
data_path = model_path = "C:/Users/Kim Jungyoon/Documents/2.study/ml/xgboost/"
train = fread(paste0(data_path, "iris.csv"))
## create dummy class
train[, c("class") := ifelse(train$species == "setosa", 1, 0)]

## define and train model
set.seed(123)
model = xgboost(data=data.matrix(train[, c("sepal_length", "sepal_width", "petal_length", "petal_width")]),
                label=train$class, verbose=0, 
                objective="binary:logistic", eta=0.3, scale_pos_weight=1,
                lambda=0.0, alpha=0, 
                max_depth=1, min_child_weight=1, tree_method="hist", nrounds=1)
## prediction on training set
preds = predict(model, data.matrix(train[, c("sepal_length", "sepal_width", "petal_length", "petal_width")]))
preds

## train data + scores
train_score = copy(train)
train_score[, c("score") := preds]

# ## fwrite
# fwrite(preds, paste0(model_path, "iris_R_xgboost_predictions.csv"))
