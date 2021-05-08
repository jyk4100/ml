## python code for <hands on ML with scikit-leran & tensor flow>
## jyk4100
## last modified 2020-08-25 

## before that... some examples with os util
## https://www.geeksforgeeks.org/python-os-path-abspath-method-with-example/
import os
os.getcwd() ## cd
os.chdir("""C:/Users/Kim Jungyoon/Documents/2.study/ml/python-ml/""") ## cd
os.listdir() ## ls
## return true if directory? hmmm
os.path.isdir( os.getcwd() )
## absoulte path vs path?
os.path.abspath('.')  
## okay...
tempdir = os.getcwd() + "\\py_prac_2020-08-24.py"
os.path.abspath(tempdir)
print(os.path.abspath(tempdir)) 

## boston housing fetching example from pg44 
## hmm this is messy...
import tarfile
from six.moves import urllib
download_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
housing_url = download_url + "datasets/housing/housing.tgz"
data_path = os.getcwd() + "\\housing.tgz"
urllib.request.urlretrieve(housing_url, data_path)
housing_tgz = tarfile.open(data_path)
housing_tgz.extractall(os.getcwd())
housing_tgz.close

##
import pandas as pd
housing = pd.read_csv(os.getcwd() + "\\housing.csv")
housing

impot matplotlib.pyplot as plt
print( housing.columns.tolist() ) ## srsly console ouput is much better in R...
## R data.table is faster than pandas...
housing.iloc[:,8].hist()

## train test split
## manual by indexing
import numpy as np
import math

def train_test_manual(data, train_ratio, randseed=123):
    shuffle_index = np.random.RandomState(seed=randseed).permutation(len(data))
    train_size = math.floor( len(data) * train_ratio )
    idx_train = shuffle_index[:train_size]
    idx_test = shuffle_index[train_size:]
    ## sprintf
    if len(idx_train) + len(idx_test) == len(data):
        print("no issue train ratio is {trts}".format(trts=round(len(idx_train)/len(data), 3)) )
        return(data.iloc[idx_train], data.iloc[idx_test])
    else:
        print("dimension doesn't match")
    ## end of ifelse... i hate indent logic end..
## end of function
hz_train, hz_test = train_test_manual(housing, 0.7)

## hash to check id?.... ## skip for now

## or use function
from sklearn.model_selection import train_test_split
train, test = train_test_split(housing, train_size=0.7, random_state=123)
