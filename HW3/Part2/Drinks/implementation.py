
# coding: utf-8

# In[49]:


import pandas as  pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


# In[50]:


data = pd.read_csv('Drinks.csv')
data.head()


# In[51]:


data.columns


# In[52]:


np.random.seed(0)

y = data[['Class 1', 'Class 2', 'Class 3']].values

# Get inputs; we define our x and y here.\n",
X = data.drop(['Class 1', 'Class 2', 'Class 3'], axis=1)
X.shape, y.shape
X = X.values


# In[53]:


def softmax(z):
    # Calculate exponent term first
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def softmax_loss(y, y_hat):
    # Clipping valu
    minval = 0.000000000001
    # Number of samples
    m = y.shape[0]
    # Loss formula, note that np.sum sums up the entire matrix and therefore does the job of two sums from the formula\n",
    loss = -1 / m * np.sum(y * np.log(y_hat.clip(min=minval)))
    return loss


# In[54]:


def loss_derivative(y, y_hat):
    return (y_hat - y)


# In[55]:


def tanh_derivative(x):
    return (1 - np.power(x, 2))


# In[56]:


# FORWARD PROPAGATION
# forward propagation function\n",
def forward_prop(model, a0):
    # Load parameters from model\n",
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']

    # Do the first Linear step \n",
    # Z1 is the input layer x times the dot product of the weights + our bias b\n",
    z1 = a0.dot(W1) + b1

    # Put it through the first activation function\n",
    a1 = np.tanh(z1)
    # Second linear step
    z2 = a1.dot(W2) + b2
    # Second activation function
    a2 = np.tanh(z2)
    # Third linear step
    z3 = a2.dot(W3) + b3

    # For the Third linear activation function we use the softmax function, either the sigmoid of softmax should be used for the last layer\n",
    a3 = softmax(z3)

    # Store all results in these values\n",
    cache = {'a0': a0, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'a3': a3, 'z3': z3}
    return cache

 


# In[57]:


#backward propagation function
def backward_prop(model, cache, y):
    # Load parameters from model
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']

    # Load forward propagation results
    a0, a1, a2, a3 = cache['a0'], cache['a1'], cache['a2'], cache['a3']

    # Get number of samples\n",
    m = y.shape[0]

    # Calculate loss derivative with respect to output
    dz3 = loss_derivative(y=y, y_hat=a3)

    # Calculate loss derivative with respect to second layer weights
    dW3 = 1 / m * (a2.T).dot(dz3)  # dW2 = 1/m*(a1.T).dot(dz2)

    # Calculate loss derivative with respect to second layer bias

    db3 = 1 / m * np.sum(dz3, axis=0)

    # Calculate loss derivative with respect to first layer
    dz2 = np.multiply(dz3.dot(W3.T), tanh_derivative(a2))

    # Calculate loss derivative with respect to first layer weights
    dW2 = 1 / m * np.dot(a1.T, dz2)

    # Calculate loss derivative with respect to first layer bias
    db2 = 1 / m * np.sum(dz2, axis=0)

    dz1 = np.multiply(dz2.dot(W2.T), tanh_derivative(a1))

    dW1 = 1 / m * np.dot(a0.T, dz1)
    db1 = 1 / m * np.sum(dz1, axis=0)

    # Store gradients
    grads = {'dW3': dW3, 'db3': db3, 'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}
    return grads

  


# In[58]:


# TRAINING PHASE


def initialize_parameters(nn_input_dim, nn_hdim, nn_output_dim):
  # First layer weights
  W1 = 2 * np.random.randn(nn_input_dim, nn_hdim) - 1

  # First layer bias
  b1 = np.zeros((1, nn_hdim))

  # Second layer weights
  W2 = 2 * np.random.randn(nn_hdim, nn_hdim) - 1

  # Second layer bias
  b2 = np.zeros((1, nn_hdim))
  W3 = 2 * np.random.rand(nn_hdim, nn_output_dim) - 1
  b3 = np.zeros((1, nn_output_dim))

  # Package and return model
  model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
  return model


def update_parameters(model, grads, learning_rate):
  # Load parameters\n",
  W1, b1, W2, b2, b3, W3 = model['W1'], model['b1'], model['W2'], model['b2'], model['b3'], model["W3"]

  # Update parameters
  W1 -= learning_rate * grads['dW1']
  b1 -= learning_rate * grads['db1']
  W2 -= learning_rate * grads['dW2']
  b2 -= learning_rate * grads['db2']
  W3 -= learning_rate * grads['dW3']
  b3 -= learning_rate * grads['db3']

  # Store and return parameters\n",
  model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
  return model


# In[59]:


def predict(model, x):
    # Do forward pass
    c = forward_prop(model, x)
    # get y_hat
    y_hat = np.argmax(c['a3'], axis=1)
    return y_hat


# In[60]:


def calc_accuracy(model, x, y):
    # Get total number of examples
    m = y.shape[0]

    # Do a prediction with the model
    pred = predict(model, x)

    # Ensure prediction and truth vector y have the same shape
    pred = pred.reshape(y.shape)

    # Calculate the number of wrong examples
    error = np.sum(np.abs(pred - y))

    # Calculate accuracy
    return (m - error) / m * 100


# In[61]:


losses = []

def train(model, X_, y_, learning_rate, epochs=20000, print_loss=False):
  # Gradient descent. Loop over epochs

  for i in range(0, epochs):

      # Forward propagation
      cache = forward_prop(model, X_)
      # a1, probs = cache['a1'],cache['a2']
      # Backpropagation

      grads = backward_prop(model, cache, y_)
      # Gradient descent parameter update
      # Assign new parameters to the model
      model = update_parameters(model=model, grads=grads, learning_rate=learning_rate)

      # Pring loss & accuracy every 100 iterations
      if print_loss and i % 100 == 0:
          a3 = cache['a3']
      print('Loss after iteration', i, ':', softmax_loss(y_, a3))
      y_hat = predict(model, X_)
      y_true = y_.argmax(axis=1)
      print('Accuracy after iteration', i, ':', accuracy_score(y_pred=y_hat, y_true=y_true) * 100, '%')
      losses.append(accuracy_score(y_pred=y_hat, y_true=y_true) * 100)
  return model
np.random.seed(0)


# In[62]:


# This is what we return at the end
model = initialize_parameters(nn_input_dim=13, nn_hdim=5, nn_output_dim=3)
model = train(model, X, y, learning_rate=0.07, epochs=5000, print_loss=True)
plt.plot(losses)   
plt.savefig('losses.png')

