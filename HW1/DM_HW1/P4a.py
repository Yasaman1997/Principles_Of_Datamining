import pandas as pd
import matplotlib.pyplot as plt


Data_Raw = pd.read_csv('iris.data',sep=',',header=-1)

plt.subplots(nrows=1, ncols=1)
plt.hist2d(list(Data_Raw.values[:,0]),list(Data_Raw.values[:,1]), bins=100)
plt.xlabel('Var1')
plt.ylabel('Var2')
plt.title('2D Histogram of data')
plt.show(block=True)