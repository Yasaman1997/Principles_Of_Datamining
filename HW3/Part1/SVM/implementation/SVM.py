
# coding: utf-8

# In[164]:


import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib as m
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import *
style.use("ggplot")
from sklearn import svm


# In[165]:


df = pd.read_csv('svmdata.csv')
df.columns = ['a', 'b','class']
df.head()


# In[166]:


x=df.drop('class',axis=1)
y=df['class']


# In[167]:


x


# In[168]:


np.array(x)


# In[169]:


y.head()


# # Split

# In[170]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=1)


# In[171]:


type(X_test)


# In[172]:


y_test.size


# In[173]:


#X_train=np.array(X_train)
#y_train=np.array(y_train)


# In[174]:


X_train


# # Plot train

# In[175]:


df.plot()


# In[176]:


df.plot.scatter(x='a',
                y='b',
                c='class',
                colormap='viridis')


# In[177]:


plt.scatter(X_train['a'], X_train['b'], c=y_train, s=50, cmap='winter')
plt.title('Train_data')
plt.savefig('train_plot.png')


# In[178]:


plt.plot(X_train,y_train) 
plt.show()


# # SVC

# In[179]:


def rms_error(model, X, y):
    y_pred = model.predict(X)
    return np.sqrt(np.mean((y - y_pred) ** 2))


# In[180]:


C=np.arange(0.01,2,0.02)
for c in C:
    clf = svm.LinearSVC(C = c, class_weight=None, dual=True, fit_intercept=True,
                        intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                        multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print("*****************")
   # print('Error:',error)
    train_accuracy = clf.score(X_train,y_train)
    print('train_accuracy:',accuracy)
    
    val_accuracy = clf.score(X_val,y_val)
    print('validation_accuracy:',accuracy)
   # print('predicted class :',y_pred)
   
    train_accuracies=list()
    train_errors=list()
    val_accuracies=list()
    val_errors=list()
    train_errors.append(clf.score(X_train,y_train))
    val_errors.append(clf.score(X_val,y_val))
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print('train_error:',train_errors)
    print('validation_error',val_errors)
 
 


# In[181]:


plt.plot(accuracies,c)


# In[182]:


accuracy.size


# # Test Evaluation
# 

# In[183]:


test_pred=clf.predict(X_test)
print("*****************")
   # print('Error:',error)
accuracy = clf.score(X_test,y_test)
print('accuracy:',accuracy)
print('predicted class :',test_pred)
   

error= rms_error(clf,X_test,y_test)
print('error:',error)
plt.plot(accuracy,c)


# In[184]:


from sklearn.metrics import classification_report, confusion_matrix 

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))  
print("*************************************************************************")
print(classification_report(y_test,y_pred))  


# In[185]:


pred.size


# In[ ]:


plt.scatter(X_test['a'], X_test['b'], c=y_pred, s=50, cmap='spring')
plt.title('Test_data')
plt.savefig('test_plot_1.png')


# In[ ]:


print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
# License: BSD 3 clause

import numpy as np
from sklearn import linear_model

# #############################################################################
# Generate sample data
n_samples_train, n_samples_test, n_features = 75, 150, 500
np.random.seed(0)
coef = np.random.randn(n_features)
coef[50:] = 0.0  # only the top 10 features are impacting the model
X = np.random.randn(n_samples_train + n_samples_test, n_features)
y = np.dot(X, coef)

# Split train and test data
X_train, X_test = X[:n_samples_train], X[n_samples_train:]
y_train, y_test = y[:n_samples_train], y[n_samples_train:]

# #############################################################################
# Compute train and test errors
alphas = np.logspace(-5, 1, 60)
enet = linear_model.ElasticNet(l1_ratio=0.7)
train_errors = list()
test_errors = list()
for alpha in alphas:
    enet.set_params(alpha=alpha)
    enet.fit(X_train, y_train)
    train_errors.append(enet.score(X_train, y_train))
    test_errors.append(enet.score(X_test, y_test))

i_alpha_optim = np.argmax(test_errors)
alpha_optim = alphas[i_alpha_optim]
print("Optimal regularization parameter : %s" % alpha_optim)

# Estimate the coef_ on full data with optimal regularization parameter
enet.set_params(alpha=alpha_optim)
coef_ = enet.fit(X, y).coef_

# #############################################################################
# Plot results functions

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.semilogx(alphas, train_errors, label='Train')
plt.semilogx(alphas, test_errors, label='Test')
plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
           linewidth=3, label='Optimum on test')
plt.legend(loc='lower left')
plt.ylim([0, 1.2])
plt.xlabel('Regularization parameter')
plt.ylabel('Performance')

# Show estimated coef_ vs true coef
plt.subplot(2, 1, 2)
plt.plot(coef, label='True coef')
plt.plot(coef_, label='Estimated coef')
plt.legend()
plt.subplots_adjust(0.09, 0.04, 1, 1, 0.26, 0.26)
plt.show()


# # NonLinear_SVC

# In[186]:


from sklearn.svm import SVC

C=np.arange(0.01,2,0.02)
for c in C:
 
    clf = SVC(kernel='rbf')
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print("*****************")
   # print('Error:',error)
    train_accuracy = clf.score(X_train,y_train)
    print('train_accuracy:',accuracy)
    
    val_accuracy = clf.score(X_val,y_val)
    print('validation_accuracy:',accuracy)
   # print('predicted class :',y_pred)
   
    train_accuracies=list()
    train_errors=list()
    val_accuracies=list()
    val_errors=list()
    train_errors.append(clf.score(X_train,y_train))
    val_errors.append(clf.score(X_val,y_val))
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print('train_error:',train_errors)
    print('validation_error',val_errors)


# In[187]:


plt.scatter(X_val['a'], X_val['b'], c=y_pred, s=50, cmap='summer')


# # test evaluation

# In[ ]:


test_pred=clf.predict(X_test)
print("*****************")
   # print('Error:',error)
accuracy = clf.score(X_test,y_test)
print('accuracy:',accuracy)
print('predicted class :',test_pred)
   

error= rms_error(clf,X_test,y_test)
print('error:',error)
plt.plot(accuracy,c)


# In[163]:


from sklearn.metrics import classification_report, confusion_matrix 

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))  
print("*************************************************************************")
print(classification_report(y_test,y_pred))  


# In[162]:


plt.scatter(X_test['a'], X_test['b'], c=y_pred, s=50, cmap='summer')
plt.title('Test_data')
plt.savefig('test_plot_2.png')


# In[ ]:


from mpl_toolkits import mplot3d

def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')

interact(plot_3D, elev=[-90, 90], azip=(-180, 180),
         X=fixed(X), y=fixed(y));

