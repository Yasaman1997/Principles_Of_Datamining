import  pandas as pd
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import math
import csv

from random import seed
from random import randrange
from csv import reader


#train=pd.read_csv('noisy_train.csv')


# Load a CSV file
data=pd.read_csv('noisy_train.csv')


def splitdataset(data):
    X = data.drop('poisonous', axis=1)
    Y = data['poisonous']
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


def Entropy(data,attribute):
    count={}

    for i in data:
        if (i[attribute] in count):
            count[i[attribute]] = count[i[attribute]] + 1
        else:
            count[i[attribute]] = 1


    e=0.0
    for j in count.values():
        e= e+ ((-1)*(j/len(data))* math.log2(j/len(data)))
    return e


def IG(data,attribute,Label):
        countSplit={}
        before_entropy=Entropy(data,attribute)
        after_entropy=0.0

        for i in data:
            if (i[attribute] in countSplit):
                countSplit[i[attribute]] = countSplit[i[attribute]] + 1
            else:
                countSplit[i[attribute]] = 1


        for value in countSplit:
            subdata = [r for r in data if (r[attribute] == value)]
            after_entropy= after_entropy + ((countSplit[value] / sum(countSplit.values())) * Entropy(subdata,Label))

        return before_entropy-after_entropy


def find_best_attribute(data,target,Label):
    data=data[:]

    for any_attr in target:
     if any_attr == target[len(target)-1]:
         continue

         information_gain=0.0
         attribute=""
         new_IG = IG(data,any_attr,target[len(target)-1])

         if (new_IG>information_gain):
             information_gain=new_IG
             attribute= any_attr

    return attribute





