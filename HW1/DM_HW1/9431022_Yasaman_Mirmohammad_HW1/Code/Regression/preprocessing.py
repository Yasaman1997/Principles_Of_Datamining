
# coding: utf-8

# In[62]:


import numpy as np


# In[63]:


a = np.load('data.npz')


# In[64]:


print(a['y'])


# In[65]:


import scipy.io
import numpy as np

data = scipy.io.loadmat("data.mat")

for i in data:
	if '__' not in i and 'readme' not in i:
		np.savetxt((""+i+".csv"),data[i],fmt='%s',delimiter=',')


# In[66]:


import pandas as pd


# In[67]:


x1=pd.read_csv('x1.csv',header=None)
x2=pd.read_csv('x2.csv',header=None)
x1_test=pd.read_csv('x1_test.csv',header=None)
x2_test=pd.read_csv('x2_test.csv',header=None)
y=pd.read_csv('y.csv',header=None)
y_test=pd.read_csv('y_test.csv',header=None)


# In[68]:


x1


# In[69]:


y


# In[70]:


y_test


# In[74]:


y=4*x2**2*x1+2*x2**2+3*x1+1


# In[75]:


plt.plot(x1,y,'b.')
plt.xlabel("$x1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
_ =plt.axis([0,2,0,15])


# In[76]:


plt.plot(x2,y,'b.')
plt.xlabel("$x2$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
_ =plt.axis([0,2,0,15])


# In[57]:


X = df[[x1,x2] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = 


# In[ ]:


# y = mx + b 
# m is slope, b is y-intercept 
def computeErrorForLineGivenPoints(b, m, points): 
totalError = 0 for i in range(0, len(points)): 
   totalError += (points[i].y — (m * points[i].x + b)) ** 2 
return totalError / float(len(points))


# # Regression

# In[48]:


np.gradient(y,2)


# In[49]:


y.shape


# In[21]:


theta = np.zeros([1,3])

#set hyper parameters
alpha = 0.01
iters = 1000


# In[22]:


cur_x = 3 # The algorithm starts at x=3
rate = 0.01 # Learning rate
precision = 0.000001 #This tells us when to stop the algorithm
previous_step_size = 1 #
max_iters = 10000 # maximum number of iterations
iters = 0 #iteration counter
df = lambda x: 2*(x+5) #Gradient of our function 


# In[23]:




# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# *Load necessary libraries*

# In[2]:


data_x = x1
data_y = y



# *Generate our data*

# In[3]:


#data_x = np.hstack((np.ones_like(data_x), data_x))


# *Add intercept data and normalize*

# In[4]:


order = np.random.permutation(len(data_x))
portion = 20
test_x = data_x[order[:portion]]
test_y = data_y[order[:portion]]
train_x = data_x[order[portion:]]
train_y = data_y[order[portion:]]


# *Shuffle data and produce train and test sets*

# In[5]:


def get_gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    sse = np.sum(np.power(error, 2))
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, sse


# *Create gradient function*

# In[6]:


w = np.random.randn(2)
alpha = 0.5
tolerance = 1e-5

# Perform Gradient Descent
iterations = 1
while True:
    gradient, error = get_gradient(w, train_x, train_y)
    new_w = w - alpha * gradient
    
    # Stopping Condition
    if np.sum(abs(new_w - w)) < tolerance:
        print ("Converged.")
        break
    
    # Print error every 50 iterations
    if iterations % 100 == 0:
        print ("Iteration: %d - Error: %.4f" %(iterations, error))
    
    iterations += 1
    w = new_w

print ("w =",w)
print ("Test Cost =", get_gradient(w, test_x, test_y)[1])


# *Perform gradient descent to learn model*

# In[9]:


plt.plot(data_x[:,1], data_x.dot(w), c='g', label='Model')
plt.scatter(train_x[:,1], train_y, c='b', label='Train Set')
plt.scatter(test_x[:,1], test_y, c='r', label='Test Set')
plt.grid()
plt.legend(loc='best')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# *Plot the model obtained*

# In[10]:


w1 = np.linspace(-w[1]*3, w[1]*3, 300)
w0 = np.linspace(-w[0]*3, w[0]*3, 300)
J_vals = np.zeros(shape=(w1.size, w0.size))

for t1, element in enumerate(w1):
    for t2, element2 in enumerate(w0):
        wT = [0, 0]
        wT[1] = element
        wT[0] = element2
        J_vals[t1, t2] = get_gradient(wT, train_x, train_y)[1]

plt.scatter(w[0], w[1], marker='*', color='r', s=40, label='Solution Found')
CS = plt.contour(w0, w1, J_vals, np.logspace(-10,10,50), label='Cost Function')
plt.clabel(CS, inline=1, fontsize=10)
plt.title("Contour Plot of Cost Function")
plt.xlabel("w0")
plt.ylabel("w1")
plt.legend(loc='best')
plt.show()


# *Generate contour plot of the cost function*

