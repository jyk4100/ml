## common R data processing functions and python pandas equivalent
## jyk4100
## last modified 2021-03-14

## pandas example code
import numpy as np
import pandas as pd
pd.__version__ ## 1.0.5 old version

## some useful function in pandas and R equivalents
iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

## dimension of data
iris.shape ## 150, 5 
## R equivalent: dim(df)

## row slice example 
iris[iris['species'] == 'setosa']['sepal_length']
## R equivalent: iris[iris$species == 'setosa,]$sepal_length

## <column>.value_counts
iris['species'].value_counts()
## R equivalent table(iris$species)

## python list comprehension useful
iris['label'] = [1 if x == 'setosa' else 0 for x in iris['species']]
iris['label'].value_counts()
## R equivalent to ifelse(iris$species == 'setosa', 1, 0)
## apply/lambda function same
iris['label2'] = iris['species'].apply(lambda x: 1 if x == "setosa" else 0)

## pd.crosstab
pd.crosstab(iris['species'], iris['label'])
pd.crosstab(iris['label'], iris['label2'])

## integer index slicing
iris.iloc[0,0] ## R iris[1,1]
iris.iloc[0,] ## R iris[1,]
iris.iloc[0,]['sepal_length'] ## R iris[1, ]$sepal_length
iris.iloc[0:4, 0:4]

## index change value in place
iris['gg'] = np.random.RandomState(123).uniform(0,1,150)
iris['gg']
iris.loc[iris['gg'] >= 0.5, 'gg'] = 0.9
## R::data.table change value inplace given condition without copying
## iris[iris$gg > 0.5, c('gg') := 0.9]

## groupby
iris.groupby('species')['label'].value_counts()
iris.groupby('species')['gg'].mean()

## namedAGG similar to R::data.table aggregation
iris.groupby('species').agg(
    mean_gg = pd.NamedAgg('gg','mean'),
    npmean_gg = pd.NamedAgg('gg', lambda x:np.nanmean(x)), 
    npmedian_gg = pd.NamedAgg('gg', lambda x:np.nanmedian(x)))

## 
def temp(gg):
    gg.sort()
    return(gg)
temp([1,2,3,9,3,7])

## basic plot compatiable without calling messed up python plots plt...
iris['gg'].plot()

import re
import numpy as np
temp = pd.DataFrame({'col1':['$1.40','$1.50', np.nan]})
temp['col1'] = temp['col1'].fillna("0")
temp['col1'] = [re.sub("[^0-9.]", "", x) for x in temp['col1']]
temp

temp = np.array([1,1,1,0,1])
temp2 = np.array([1,0,0,0,0])