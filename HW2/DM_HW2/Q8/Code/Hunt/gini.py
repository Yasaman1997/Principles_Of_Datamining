import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


train=pd.read_csv('noisy_train.csv')

X_1 = train.drop('poisonous', axis=1)
Y_1 = train['poisonous']


# Spliting the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1, test_size=0.2, random_state=100)

clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


clf_gini.predict([[4, 2, 1, 3]])

y_pred = clf_gini.predict(X_test)
y_pred


y_pred_en = clf_entropy.predict(X_test)
y_pred_en


print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)

print ("Accuracy is ", accuracy_score(y_test,y_pred_en)*100)