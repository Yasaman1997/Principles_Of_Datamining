#-*- coding: utf-8 -*-
__author__ = 'PF'
import numpy
import csv
import matplotlib.pyplot as plt
Feature_number=4#檔案的Feature種類數量
Training_number=50#每個CLASS中個別Feature的數量
CUS_NUMBER=50#使用者自訂 想要挑選的Feature數
all_Feature=False#是否要全部的資料都計算
Iris_setosa=[]
Iris_versicolor=[]
Iris_virginica=[]
Iris=[]
label=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
def get_covariance_matrix(A):
    if all_Feature == False:
        number=CUS_NUMBER
    else:
        number=Training_number
    A=numpy.reshape(A,(number,Feature_number))#將一維矩陣轉換成50*Feature_number的矩陣
    A=numpy.array(A, dtype='f')#把這個矩陣內的值規定為浮點數
    mean_vector=get_mean_vector(A)#呼叫MEAN_VECTOR
    cov_matrix = numpy.reshape(numpy.zeros(Feature_number*Feature_number), (Feature_number,Feature_number))#先創全為0的
#將原本的矩陣減掉MEAN_VECTOR(第X列的所有元素-第Y列的均值)
    for x in range(Feature_number):
        for y in range(len(A[:,x])):
            A[:,x][y]=float(A[:,x][y])-float(mean_vector[x])
#求協方差(i,j)=（第X列的所有元素-第X列的均值）*（第X列的所有元素-第Y列的均值）
#相減的部份上一個FOR迴圈已經處理過了,這邊就是個矩陣乘法
    for x in range(Feature_number):
        for y in range(Feature_number):
            dot=0
            for z in range(len(A[:,x])):
                dot=float(A[:,x][z])*float(A[:,y][z])+dot#X列＊Y列
            cov_matrix[x][y]=dot/(number-1)#存回COV_MATRIX,比照MALTLAB方法除以一個總數-1
    print(cov_matrix)
def get_mean_vector(A):
    mean_vector=[]
    for i in range(Feature_number):
        sum=0
        for value in A[:,i]:
            sum=sum+float(value)#將第I列的元素壘加起來
        mean_vector.append(float(sum/len(A[:,i])))#將平均值加入MEAN_VECTOR的一維陣列中
    return mean_vector


def data_processing():
    X=-1
    fn=open("HW1/iris.data.txt","r")#開檔
    for row in csv.DictReader(fn,label):#將檔案用CSV方式讀入並且給予標籤方便操作
        X=X+1
        for i in range(Feature_number):#依照CLASS將四個特徵直都放入個別陣列中
            Iris.append(row[label[i]])#將所有的值存入IRIS的陣列
            if str(row["class"]) == "Iris-setosa":#將所有的值存入IRIS_SETOSA的陣列
                if all_Feature== True:
                    Iris_setosa.append(row[label[i]])
                else:
                    if X%(Training_number/CUS_NUMBER)==0 and len(Iris_setosa)<CUS_NUMBER*4:
                        Iris_setosa.append(row[label[i]])
            elif str(row["class"]) == "Iris-versicolor":#將所有的值存入IRIS_VERSICOLOR的陣列
                if all_Feature== True:
                    Iris_versicolor.append(row[label[i]])
                else:
                    if X%(Training_number/CUS_NUMBER)==0 and len(Iris_versicolor)<CUS_NUMBER*4:
                        Iris_versicolor.append(row[label[i]])
            else:#將所有的值存入最後一個IRIS的陣列
                    if all_Feature== True:
                        Iris_virginica.append(row[label[i]])
                    else:
                        if X%(Training_number/CUS_NUMBER)==0 and len(Iris_virginica)<CUS_NUMBER*4:
                            Iris_virginica.append(row[label[i]])
    fn.close()#關檔
def draw():
    for m in range(Feature_number):
        for n in range(Feature_number):
            if m < n:
                fn=open("HW1/iris.data.txt","r")
                for row in csv.DictReader(fn, label):
                    #plt.xlim(0,10)
                    #plt.ylim(0,10)
                    plt.xlabel(label[m])#LABEL設定
                    plt.ylabel(label[n])#LABEL設定
                    plt.title(label[m]+"  and  "+label[n])#TITLE設定
                    x = row[label[m]]#X值
                    y = row[label[n]]#Y值
                    if row["class"]=="Iris-setosa":
                        plt.plot(x,y,"ro")#Iris_setosa用紅色點表示
                    elif row["class"]=="Iris-versicolor":
                        plt.plot(x,y,"bo")#Iris_versicolor用藍色點表示
                    else:
                        plt.plot(x,y,"go")#Iris_virginica用綠色點表示
                plt.savefig("HW1/"+label[m]+"_and_"+label[n]+".png",dpi=300,format="png")
                plt.close()#關圖,清緩存
                fn.close()#關檔


if __name__ == "__main__":#主程式
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

    print("Iris_setosa mean vector\n")
    print(get_mean_vector(numpy.reshape(Iris_setosa,(number,Feature_number))))
    print("Iris_versicolor mean vector\n")
    print(get_mean_vector(numpy.reshape(Iris_versicolor,(number,Feature_number))))
    print("Iris_virginica mean vector\n")
    print(get_mean_vector(numpy.reshape(Iris_virginica,(number,Feature_number))))
    draw()
