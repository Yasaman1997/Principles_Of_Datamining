import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

data=pd.read_csv('US Presidential Data.csv')

data.head()


# create training and testing vars

#train test split
x=data.drop('Win/Loss',axis=1)
y=data['Win/Loss']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)

train_set=(X_train,y_train)
test_set=(X_test,y_test)


#Normalization
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#knn with neighbor 1(NN1)
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(classifier.score(X_test,y_test))

c_error=(((y_pred - y_test) ** 2).sum()) / len(y_pred)
print('Classification Error:',c_error)


##error claculation based on k value
error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    prediction=knn.predict(X_test)
    error.append(np.mean(prediction != y_test))


#plot
plt.figure(figsize=(12, 12))


plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()





