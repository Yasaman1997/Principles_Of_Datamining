import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def Gradient_Descent(Input, Target, Weights, Learning_Rate, Dimen, Iter):
    Input_Transpose = Input.transpose()
    SSE_Array = np.zeros(Iter)
    for in_Iter in range(0, Iter):
        Predicted_Target = np.dot(Input, Weights)
        Error = Predicted_Target - Target
        SSE = np.sum(Error ** 2)
        gradient = np.dot(Input_Transpose, Error) / Dimen
        Weights = Weights - Learning_Rate * gradient
        SSE_Array[in_Iter] = SSE
        print("Iter %d with SSE: %f" % (in_Iter, SSE))
    return Weights, SSE_Array


# Main Code
Data_Raw = np.load('data.npz')
x1 = Data_Raw.f.x1
x1_test = Data_Raw.f.x1_test
x2 = Data_Raw.f.x2
x2_test = Data_Raw.f.x2_test
y = Data_Raw.f.y
y_test = Data_Raw.f.y_test

Gradien_Order = 1
Bias_Train = np.ones([np.shape(x1)[0],Gradien_Order])
Bias_Test = np.ones([np.shape(x1_test)[0],Gradien_Order])
x_train = np.column_stack((np.multiply(x1,x2 **2),x2 **2,x1,Bias_Train))
x_test = np.column_stack((np.multiply(x1_test,x2_test **2),x2_test **2,x1_test,Bias_Test))

Sample_Size, Dimen = np.shape(x_train)
Iter= 1000
Learning_Rate = 0.0000001
Weights = np.ones(Dimen)
Weights, SSE_Array = Gradient_Descent(x_train, y, Weights, Learning_Rate, Sample_Size, Iter)

y_p_test = Weights[0]*x_test[:,0] + Weights[1]*x_test[:,1] + Weights[2]*x_test[:,2]+ Weights[3]*x_test[:,3]
y_p_train = Weights[0]*x_train[:,0] + Weights[1]*x_train[:,1] + Weights[2]*x_train[:,2]+ Weights[3]*x_train[:,3]

Error = y_p_test - y_test
SSE_Test = np.sum(Error ** 2)
Error = y_p_train - y
SSE_Train = np.sum(Error ** 2)

print("SSE > Test = %f   &  Train =  %f " % (SSE_Test,SSE_Train))

# Plot Target and Perdicted Target
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x1_test, x2_test, y_test , c='red')
ax.scatter(x1_test, x2_test, y_p_test , c='blue')
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylabel('y')
ax.legend(['Target','Prediction'])

# Plot Train SSE line
fig = plt.figure()
plt.plot(range(0,Iter),SSE_Array,c='red')
plt.xlabel('Iteration')
plt.ylabel('SSE')
plt.legend(['Train SSE Line'])
plt.show()