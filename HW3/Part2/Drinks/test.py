
# coding: utf-8

# In[18]:


import pandas as  pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow
from keras import layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy


# In[19]:


data = pd.read_csv('Drinks.csv')
data.head()


# In[20]:


import tensorflow as tf
import keras.layers

n_neurons_h = 178
n_neurons_out = 3
n_epochs = 4500
learning_rate = 0.7

model = tf.keras.Sequential()
model.add(layers.Dense(n_neurons_h, activation="tanh"))
model.add(layers.Dense(n_neurons_h, activation="tanh"))
model.add(layers.Dense(n_neurons_out, activation="softmax"))



model.fit(training_data, training_labels, epochs=n_epochs, batch_size=32)

model.compile(optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),loss="binary_crossentropy", metrics=["accuracy"])
model.fit(training_X, training_y, epochs=n_epochs)

