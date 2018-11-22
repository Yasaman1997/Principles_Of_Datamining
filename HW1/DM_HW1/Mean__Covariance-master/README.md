
[![DUB](https://img.shields.io/dub/l/vibe-d.svg)]()

# **Mean Vector & Covariance Matrix** 
This is course homework project No.1 on Spring 2015 pattern recognition at CS, NCHU.
## **Issue**
Calculate the mean vector and covariance of three class data in [Iris Dataset](http://archive.ics.uci.edu/ml/), get form UCI Machine Learning Repository,  Iris_setosa, Iris_versicolor and Iris_virginica.
## **Dataset**
The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres.

## **Mean Vector**
The mean vector consists of the means of each variable as following:

![mean](https://github.com/nightheronry/Mean__Covariance/blob/master/Mean.png)

```python
def get_mean_vector(A):
    mean_vector=[]
    for i in range(Feature_number):
        sum=0
        for value in A[:,i]:
            sum=sum+float(value)#accumulate all element in row i
        mean_vector.append(float(sum/len(A[:,i])))#add average value to MEAN_VECTOR
    return mean_vector
```
## **covariance matrix**
The variance-covariance matrix consists of the variances of the variables along the main diagonal and the covariances between each pair of variables in the other matrix positions.
The formula for computing the covariance of the variables _X_ and _Y_ is

![covariance](https://github.com/nightheronry/Mean__Covariance/blob/master/covariance.png)

```python
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
``` 
## **Result**
Data Number: 50
- Iris_setosa
```sh
mean_vector: 
[5.005999999999999, 3.4180000000000006, 1.464, 0.2439999999999999]
get_covariance_matrix:
[[ 0.12424897  0.10029795  0.01613878  0.01054694]
 [ 0.10029795  0.14517959  0.01168164  0.01143674]
 [ 0.01613878  0.01168164  0.03010613  0.00569796]
 [ 0.01054694  0.01143674  0.00569796  0.01149388]]
```
- Iris_versicolor
```sh
mean_vector: 
[5.936, 2.7700000000000005, 4.26, 1.3259999999999998]
get_covariance_matrix: 
[[ 0.26643266  0.08518367  0.18289797  0.05577959]
 [ 0.08518367  0.09846939  0.08265305  0.04120408]
 [ 0.18289797  0.08265305  0.22081632  0.07310204]
 [ 0.05577959  0.04120408  0.07310204  0.03910612]]
```
- Iris_virginica
```sh
mean_vector: 
[6.587999999999998, 2.9739999999999998, 5.552, 2.026]
get_covariance_matrix: 
[[ 0.40434278  0.09376325  0.30328976  0.04909387]
 [ 0.09376325  0.10400408  0.07137958  0.04762857]
 [ 0.30328976  0.07137958  0.30458773  0.04882448]
 [ 0.04909387  0.04762857  0.04882448  0.07543266]]
``` 
Considering the two features, sepal_length and sepal_width (mean_vector[0] and mean_vector[1]), we find Iris_setosa(Red) is
 far from the others. By contrast, Iris_versicolor(Blue) and Iris_virginica(Green) are near each other.


![dataplot](https://github.com/nightheronry/Mean__Covariance/blob/master/dataplot.png)

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).
