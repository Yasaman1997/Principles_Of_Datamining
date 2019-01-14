import pandas as  pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv('Drinks.csv')
print(data.head())

np.random.seed(0)

y = data['Class 1', 'Class 2', 'Class 3'].values

# Get inputs; we define our x and y here.\n",
X = data.drop(['Class1', 'Class 2', 'Class 3'], axis=1)
X.shape, y.shape
X = X.values
