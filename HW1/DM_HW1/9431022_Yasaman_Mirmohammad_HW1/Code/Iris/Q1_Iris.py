
# coding: utf-8

# # Iris

# In[298]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[299]:


import numpy as np # linear algebra
import seaborn as sns
sns.set(style='whitegrid')
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pylab
import seaborn as sns
import numpy as np
from IPython.core.display import display, HTML


from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[300]:


#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# # 0: Read the data 

# In[301]:


iris = pd.read_csv('Iris.csv')
iris=iris.drop('Id', axis=1)


# In[302]:


iris.shape


# In[303]:


iris.head(10)


# In[304]:


iris.describe()


# In[305]:


iris.isnull().sum()


# In[306]:


# box and whisker plots
iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# # D:  Numerical processing

# In[307]:


#findout no of rows for each Species.
print(iris.groupby('Species').size())


# In[308]:


iris.head()


# In[309]:


stats=pd.DataFrame()
stats["mean"]=iris.mean()
stats["Var"]=iris.var()
stats


# In[310]:


iris.describe()


# # F:Correlation

# In[330]:


iris.corr()


# In[331]:


def get_mean_vector(A):
    mean_vector=[]
    for i in range(Feature_number):
        sum=0
        for value in A[:,i]:
            sum=sum+float(value)#accumulate all element in row i
        mean_vector.append(float(sum/len(A[:,i])))#add average value to MEAN_VECTOR
    return mean_vector


# In[332]:


setosa_mean=setosa.mean()
versicolor_mean=versicolor.mean()
virginica_mean=virginica.mean()


# In[333]:


np.corrcoef(setosa_mean,versicolor_mean)


# In[334]:


np.corrcoef(setosa_mean,virginica_mean)


# In[335]:


np.corrcoef(versicolor_mean,virginica_mean)


# # E:Covariance

# In[336]:


iris.cov()


# In[337]:


def get_mean_vector(A):
    mean_vector=[]
    for i in range(Feature_number):
        sum=0
        for value in A[:,i]:
            sum=sum+float(value)#accumulate all element in row i
        mean_vector.append(float(sum/len(A[:,i])))#add average value to MEAN_VECTOR
    return mean_vector


# In[338]:


def get_covariance_matrix(A):
    if all_Feature == False:
        number=CUS_NUMBER
    else:
        number=Training_number
    A=numpy.reshape(A,(number,Feature_number))#transform One-dimensional matrix to matrix50*Feature_number matrix
    A=numpy.array(A,dtype='f')#set the values in the array are float
    mean_vector=get_mean_vector(A)#call MEAN_VECTOR()
    cov_matrix = numpy.reshape(numpy.zeros(Feature_number*Feature_number), (Feature_number,Feature_number))#matrix initialize
#original matrix minus MEAN_VECTOR
    for x in range(Feature_number):
        for y in range(len(A[:,x])):
            A[:,x][y]=float(A[:,x][y])-float(mean_vector[x])
#covariance(i,j)
#matrix multiply
    for x in range(Feature_number):
        for y in range(Feature_number):
            dot=0
            for z in range(len(A[:,x])):
                dot=float(A[:,x][z])*float(A[:,y][z])+dot#row_x＊row_Y
            cov_matrix[x][y]=dot/(number-1)#storage back to COV_MATRIX,them divide by N-1
    print(cov_matrix)


# In[339]:


import numpy
import csv
import matplotlib.pyplot as plt



all_Feature=False
Iris_setosa=[]
Iris_versicolor=[]
Iris_virginica=[]
Iris=[]


Feature_number=4
Training_number=50
CUS_NUMBER=50

label=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]



def data_processing():
    X=-1
    fn=open("iris.data.txt","r")
    for row in csv.DictReader(fn,label):
        X=X+1
        for i in range(Feature_number):
            Iris.append(row[label[i]])
            if str(row["class"]) == "Iris-setosa":
                if all_Feature== True:
                    Iris_setosa.append(row[label[i]])
                else:
                    if X%(Training_number/CUS_NUMBER)==0 and len(Iris_setosa)<CUS_NUMBER*4:
                        Iris_setosa.append(row[label[i]])
            elif str(row["class"]) == "Iris-versicolor":
                if all_Feature== True:
                    Iris_versicolor.append(row[label[i]])
                else:
                    if X%(Training_number/CUS_NUMBER)==0 and len(Iris_versicolor)<CUS_NUMBER*4:
                        Iris_versicolor.append(row[label[i]])
            else:
                    if all_Feature== True:
                        Iris_virginica.append(row[label[i]])
                    else:
                        if X%(Training_number/CUS_NUMBER)==0 and len(Iris_virginica)<CUS_NUMBER*4:
                            Iris_virginica.append(row[label[i]])
    fn.close()
    
    



# In[340]:


def draw():
    for m in range(Feature_number):
        for n in range(Feature_number):
            if m < n:
                fn=open("iris.data.txt","r")
                for row in csv.DictReader(fn, label):
                  
                    plt.xlabel(label[m])#
                    plt.ylabel(label[n])
                    plt.title(label[m]+"  and  "+label[n])
                    x = row[label[m]]#X
                    y = row[label[n]]#Y
                    if row["class"]=="Iris-setosa":
                        plt.plot(x,y,"ro")#setosa
                    elif row["class"]=="Iris-versicolor":
                        plt.plot(x,y,"bo")#versicolor
                    else:
                        plt.plot(x,y,"go")#virginica
                plt.savefig(""+label[m]+"_and_"+label[n]+".png",dpi=300,format="png")
                plt.close()
                fn.close()


# In[341]:


data_processing()
print("Iris_setosa\n")
get_covariance_matrix(Iris_setosa)
print("Iris_versicolor\n")
get_covariance_matrix(Iris_versicolor)
print("Iris_virginica\n")
print(get_covariance_matrix(Iris_virginica))

if all_Feature == False:
    number=CUS_NUMBER
else:
    number=Training_number
print("Data number: "+str(number)+"\n")

print("Iris_setosa mean vector:\n")
print(get_mean_vector(numpy.reshape(Iris_setosa,(number,Feature_number))))
print("\n")
print("Iris_versicolor mean vector:\n")
print(get_mean_vector(numpy.reshape(Iris_versicolor,(number,Feature_number))))
print("\n")
print("Iris_virginica mean vector:\n")
print(get_mean_vector(numpy.reshape(Iris_virginica,(number,Feature_number))))
print("\n")
  


# In[343]:


draw()


# # Dataframe

# In[ ]:


#Create 3 DataFrame for each Species
setosa=iris[iris['Species']=='Iris-setosa']
versicolor =iris[iris['Species']=='Iris-versicolor']
virginica =iris[iris['Species']=='Iris-virginica']

print('setosa')
print(setosa.describe())
print("*********************************************************************************")
print('versicolor')
print(versicolor.describe())
print("*********************************************************************************")
print('virginica')
print(virginica.describe())


# # G:Virginica

# In[ ]:


virginica.hist()
plt.show()


# In[ ]:


virginica.corr()


# In[ ]:


virginica.cov()


# In[ ]:


species_list = list(iris['Species'].unique())
print("Types of species: %s\n" % species_list)

print("Dataset length: %i\n" % len(iris))

print("Sepal length range: [%s, %s]" % (min(iris["SepalLengthCm"]), max(iris["SepalLengthCm"])))
print("Sepal width range:  [%s, %s]" % (min(iris["SepalWidthCm"]), max(iris["SepalLengthCm"])))
print("Petal length range: [%s, %s]" % (min(iris["PetalLengthCm"]), max(iris["PetalLengthCm"])))
print("Petal width range:  [%s, %s]\n" % (min(iris["PetalWidthCm"]), max(iris["PetalWidthCm"])))

print("Sepal length variance:\t %f" % np.var(iris["SepalLengthCm"]))
print("Sepal width variance: \t %f" % np.var(iris["SepalWidthCm"]))
print("Petal length variance:\t %f" % np.var(iris["PetalLengthCm"]))
print("Petal width variance: \t %f\n" % np.var(iris["PetalWidthCm"]))

print("Sepal length stddev:\t %f" % np.std(iris["SepalLengthCm"]))
print("Sepal width stddev: \t %f" % np.std(iris["SepalWidthCm"]))
print("Petal length stddev:\t %f" % np.std(iris["PetalLengthCm"]))
print("Petal width stddev: \t %f\n" % np.std(iris["PetalWidthCm"]))

print("Description\n---")
print(iris[iris.columns[2:]].describe())


# # A:2D Histogram

# In[ ]:


plt.hist2d(iris.SepalLengthCm, iris.SepalWidthCm, bins=50, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')


# In[ ]:


plt.hist2d(iris.PetalLengthCm, iris.PetalWidthCm, bins=50, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.xlabel('PetalLength')
plt.ylabel('PetalWidth')


# In[ ]:


plt.hist2d(iris.SepalLengthCm, iris.PetalWidthCm, bins=50, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.xlabel('SepalLength')
plt.ylabel('PetalWidth')


# In[ ]:


plt.hist2d(iris.SepalLengthCm, iris.PetalLengthCm, bins=50, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.xlabel('SepalLength')
plt.ylabel('PetalLengthCm')


# In[ ]:


plt.hist2d(iris.SepalWidthCm, iris.PetalWidthCm, bins=50, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
plt.xlabel('SepalWidthCm,')
plt.ylabel('PetalWidthCm,')


# In[ ]:


iris.hist()
plt.show()


# In[ ]:


g = sns.pairplot(iris, hue='Species', markers='+')
plt.show()


# # B:# 3D Histogram

# In[ ]:


iris.head(10)


# In[ ]:


#Sepal length
X[:, 1] 


# In[ ]:


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  

iris = datasets.load_iris()
X = iris.data[:, :2]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = X[:, 0] #Sepal Length
y = X[:, 1]  #SepalWidth
hist, xedges, yedges = np.histogram2d(x, y)

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(iris.data)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [iris["PetalWidthCm"], iris["PetalLengthCm"]]
n = 100
ax.scatter(iris["PetalLengthCm"], iris["PetalWidthCm"],iris["SepalWidthCm"])


ax.set_xlabel('PetalLengthCm')
ax.set_ylabel('PetalWidthCm')
ax.set_zlabel('SepalWidthCm')

plt.tight_layout(pad=0.5)
plt.show()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [iris["PetalWidthCm"], iris["PetalLengthCm"]]
n = 100
ax.scatter(iris["PetalLengthCm"], iris["PetalWidthCm"],iris["SepalLengthCm"])


ax.set_xlabel('PetalLengthCm')
ax.set_ylabel('PetalWidthCm')
ax.set_zlabel('SepalLengthCm')

plt.tight_layout(pad=0.5)
plt.show()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [iris["PetalWidthCm"], iris["PetalLengthCm"]]
n = 100
ax.scatter(iris["PetalWidthCm"], iris["SepalLengthCm"],iris["SepalLengthCm"])


ax.set_xlabel('PetalWidthCm')
ax.set_ylabel('SepalLengthCm')
ax.set_zlabel('SepalLengthCm')

plt.tight_layout(pad=0.5)
plt.show()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [iris["PetalWidthCm"], iris["PetalLengthCm"]]
n = 100
ax.scatter(iris["PetalLengthCm"], iris["SepalLengthCm"],iris["SepalWidthCm"])


ax.set_xlabel('PetalLengthCm')
ax.set_ylabel('SepalLengthCm')
ax.set_zlabel('SepalWidthCm')

plt.tight_layout(pad=0.5)
plt.show()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [iris["PetalWidthCm"], iris["PetalLengthCm"]]
n = 100
ax.scatter(iris["PetalLengthCm"], iris["SepalLengthCm"],iris["SepalLengthCm"])


ax.set_xlabel('PetalLengthCm')
ax.set_ylabel('SepalLengthCm')
ax.set_zlabel('SepalLengthCm')

plt.tight_layout(pad=0.5)
plt.show()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = [iris["PetalWidthCm"], iris["PetalLengthCm"]]
n = 100
ax.scatter(iris["PetalLengthCm"], iris["SepalWidthCm"],iris["SepalLengthCm"])


ax.set_xlabel('PetalLengthCm')
ax.set_ylabel('SepalWidthCm')
ax.set_zlabel('SepallengthCm')

plt.tight_layout(pad=0.5)
plt.show()


# In[ ]:


from sklearn import datasets
import pandas as pd

# Load some data
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
iris_df['species'] = iris['target']

colours = ['red', 'orange', 'blue']
species = ['I. setosa', 'I. versicolor', 'I. virginica']

for i in range(0, 3):    
    species_df = iris_df[iris_df['species'] == i]    
    plt.scatter(        
        species_df['sepal length (cm)'],        
        species_df['petal length (cm)'],
        color=colours[i],        
        alpha=0.5,        
        label=species[i]   
    )

plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')
plt.title('Iris dataset: petal length vs sepal length')
plt.legend(loc='lower right')

plt.show()


# In[ ]:


display(HTML('<h1>Analyzing the ' +
             '<a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient">' +
             'Pearson correlation coefficient</a></h1>'))

# data without the indexes
dt = iris[iris.columns]

# method : {‘pearson’, ‘kendall’, ‘spearman’}
corr = dt.corr(method="pearson") #returns a dataframe, so it can be reused

# eliminate upper triangle for readability
bool_upper_matrix = np.tril(np.ones(corr.shape)).astype(np.bool)
corr = corr.where(bool_upper_matrix)
display(corr)
# alternate method: http://seaborn.pydata.org/examples/many_pairwise_correlations.html

# seaborn matrix here
#sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#            square=True, ax=ax)
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# # H:Feature Selection

# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


X=iris.drop('Species',axis=1)
y=iris.Species
X.head()


# In[ ]:


iris=pd.read_csv('Iris.csv')

# To make a Pandas boxplot grouped by species, use .boxplot
#Modify the figsize, by placing a value in the X and Y cordinates
iris.drop("Id", axis=1).boxplot(by="Species", figsize=(10, 10))
plt.show()


# In[ ]:


X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new


# In[ ]:


clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new              


#  Best Features: PetalWidth + PetalWidth

# In[ ]:


clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
clf.feature_importances_  

model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
X_new              


# In[ ]:


iris.Species = iris.Species.replace(to_replace=['Iris-setosa', 'Iris-versicolor','Iris-virginica'], value=[0, 1,2])


# # Validation Set

# In[ ]:


# Split-out validation dataset
array = iris.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Test Harness:
# We will use 10-fold cross validation to estimate accuracy.
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.

# We are using the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.

# We are going to test the following algorithms to know which one is the best to to take care of our data set:
# 
# Logistic Regression (LR)
# Linear Discriminant Analysis (LDA)
# K-Nearest Neighbors (KNN).
# Classification and Regression Trees (CART).
# Gaussian Naive Bayes (NB).
# Support Vector Machines (SVM).
# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms.

# In[ ]:


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = model_selection.KFold(n_splits=10, random_state=seed)
 cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)


# Then we’ll choose the best algorithm: KNN seems to be the best with the value 0.983

# In[ ]:


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# # Split data + Models
# 
# * trainset: 80%
# * testset: 20%

# In[ ]:


# set seed for numpy and tensorflow
# set for reproducible results
seed = 5
np.random.seed(seed)
tf.set_random_seed(seed)


# In[ ]:


# set replace=False, Avoid double sampling
train_index = np.random.choice(len(X), round(len(X) * 0.8), replace=False)


# In[ ]:


# diff set
test_index = np.array(list(set(range(len(X))) - set(train_index)))
train_X = X[train_index]
train_y = y[train_index]
test_X = X[test_index]
test_y = y[test_index]


# In[ ]:


# Define the normalized function
def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)


# # Normalized processing

# In[ ]:


# Normalized processing, must be placed after the data set segmentation, 
# otherwise the test set will be affected by the training set
train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)

#iris.Species = iris.Species.replace(to_replace=['Iris-setosa', 'Iris-versicolor'], value=[0, 1])

