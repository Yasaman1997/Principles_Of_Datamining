import pandas as pd
from sklearn.model_selection import train_test_split
from csv import reader
from sys import exit
from math import sqrt
from operator import itemgetter
import math

data = pd.read_csv('US Presidential Data.csv')

x = data.drop('Win/Loss', axis=1)
y = data['Win/Loss']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

train_set = (X_train, y_train)
test_set = (X_test, y_test)


def load_data_set(filename):
    try:
        with open(filename, newline='') as data:
            return list(reader(data, delimiter=','))
    except FileNotFoundError as e:
        raise e


def convert_to_float(data_set, mode):
    new_set = []
    try:
        if mode == 'training':
            for data in data_set:
                new_set.append([float(x) for x in data[:len(data) - 1]] + [data[len(data) - 1]])

        elif mode == 'test':
            for data in data_set:
                new_set.append([float(x) for x in data])

        else:
            print('Invalid mode, program will exit.')
            exit()

        return new_set

    except ValueError as v:
        print(v)
        print('Invalid data set format, program will exit.')
        exit()


def get_classes(train_set):
    return list(set([c[-1] for c in training_set]))


def find_neighbors(distances, k):
    return distances[0:k]


def find_response(neighbors, classes):
    votes = [0] * len(classes)

    for instance in neighbors:
        for ctr, c in enumerate(classes):
            if instance[-2] == c:
                votes[ctr] += 1

    return max(enumerate(votes), key=itemgetter(1))


def knn(train_set, test_set, k):
    distances = []
    dist = 0
    limit = len(training_set[0]) - 1

    # generate response classes from training data
    classes = get_classes(training_set)

    try:
        for test_instance in test_set:
            for row in training_set:
                for x, y in zip(row[:limit], test_instance):
                    dist += (x - y) * (x - y)
                distances.append(row + [sqrt(dist)])
                dist = 0

            distances.sort(key=itemgetter(len(distances[0]) - 1))

            # find k nearest neighbors
            neighbors = find_neighbors(distances, k)

            # get the class with maximum votes
            index, value = find_response(neighbors, classes)

            # Display prediction
            print('The predicted class for sample ' + str(test_instance) + ' is : ' + classes[index])
            print('Number of votes : ' + str(value) + ' out of ' + str(k))

            # empty the distance list
            distances.clear()

    except Exception as e:
        print(e)


def main():
    try:
        # get value of k
        k = int(input('Enter the value of k : '))

        # load the training and test data set

        # training_file = input('Enter name of training data file : ')
        # test_file = input('Enter name of test data file : ')
        # create training and testing vars

        train_set = convert_to_float(load_data_set(train_set), 'training')
        test_set = convert_to_float(load_data_set(test_file), 'test')

        if not train_set:
            print('Empty training set')

        elif not test_set:
            print('Empty test set')

        elif k > len(train_set):
            print('Expected number of neighbors is higher than number of training data instances')

        else:
            knn(train_set, test_set, k)

    except ValueError as v:
        print(v)

    except FileNotFoundError:
        print('File not found')


if __name__ == '__main__':
    main()
