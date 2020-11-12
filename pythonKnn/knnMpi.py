from collections import Counter
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import sys

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        valid = self.k <= len(X)
        if not valid:
            raise Exception('The value of k should be less than the size of training data')

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels

    def _predict(self, x):
        # compute distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[0:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common label
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common_label

    def getLocalSortedDistances(self, X):
        k_local_sorted_distances = [self._getLocalSortedDistances(x) for x in X]
        return k_local_sorted_distances

    def _getLocalSortedDistances(self, x):
        # array of tuples (distance, label)
        distances = []
        for i in range(0, len(self.X_train)):
          distance = (euclidean_distance(x, self.X_train[i]), self.y_train[i])
          distances.append(distance)
        # sort the array in ascending order by distance and pick k first elements 
        k_local_sorted_distances = sorted(distances, key=lambda t: t[0])[0:self.k]
        return k_local_sorted_distances

def euclidean_distance(p, q):
    return np.sqrt(np.sum((p-q)**2))

def load_data():
    # load data from csv files
    training_data = pd.read_csv('https://gitlab.com/duyvv/parallel-programming-project/raw/master/data/credit_train.csv')
    test_data = pd.read_csv('https://gitlab.com/duyvv/parallel-programming-project/raw/master/data/credit_test.csv')
    return training_data, test_data

def extract_data(data_records, training_data, test_data):
    # Extract a number of records according to testing cases
    X_train = training_data.iloc[:data_records,:(training_data.shape[1]-1)]
    y_train = training_data.iloc[:data_records,(training_data.shape[1]-1)]
    X_test = test_data.iloc[:500,:(test_data.shape[1]-1)]
    y_test = test_data.iloc[:500,(test_data.shape[1]-1)]  
    # transform into numpy array for faster computation
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test

def allocData(path, nAttr):
    trainSize = sum(1 for line in open(path))
    x_data = np.empty((0, nAttr), float)
    y_data = np.empty((0, ), float)

    f = open(path, "r")
    for line in f:
        row = [float(x) for x in line.split(" ")]
        attrs = row[:-1]
        label = row[-1]

        x_data = np.vstack([x_data, attrs])
        y_data = np.append(y_data, label)

    f.close()

    return x_data, y_data

#----------- KNN --------------

def parallel_KNN(k, X_train, y_train, X_test, y_test):
    # MPI handling
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    tag_local_result = 1000
    tag_x_train = 1001
    tag_y_train = 1002
    masterID = 0

    # Master process
    if rank == masterID:
        start = MPI.Wtime()

        # 1. divide training data by number of processes
        train_data_size = len(X_train)
        average_records_per_process = int(train_data_size / num_processes)
        # print('Average records per process = '+ str(average_records_per_process))
        for processID in range(1,num_processes):
            start_index = average_records_per_process * processID
            end_index = average_records_per_process * (processID + 1)
            X_train_portion = X_train[start_index : end_index]
            y_train_portion = y_train[start_index : end_index]
            # print(X_train_portion.shape, y_train_portion.shape, start_index, end_index)
            comm.send(X_train_portion, dest=processID, tag=tag_x_train)
            comm.send(y_train_portion, dest=processID, tag=tag_y_train)

        # 2. consider the master also as a worker, find the first local kNN
        X0_train_portion = X_train[0:average_records_per_process]
        y0_train_portion = y_train[0:average_records_per_process]
        kNN = KNN(k)
        kNN.fit(X0_train_portion, y0_train_portion)
        global_sorted_distances = kNN.getLocalSortedDistances(X_test)

        # 3. collect local results from slave processes
        for processID in range(1,num_processes):
            status = MPI.Status()
            local_result = comm.recv(source=MPI.ANY_SOURCE, tag=tag_local_result, status=status)
            # print('Result from processs ID = ' + str(status.Get_source()))
            # print(local_result)
            global_sorted_distances = np.concatenate((global_sorted_distances, local_result), axis=1) # column-wise concat

        # 4. globally select k nearest neighbors and do major voting
        global_knn = np.array([sorted(element, key=lambda t: t[0])[0:k] for element in global_sorted_distances])
        most_common_labels = [(Counter(element[:,1]).most_common(1)[0][0]) for element in global_knn]
        accuracy = accuracy_score(y_test, most_common_labels)
        #print('Accuracy = ' + str(accuracy))

        end = MPI.Wtime()
        elapsed_time = end - start
        #if num_processes == 1:
            #print('Serial Elapsed time = ' + str(elapsed_time))
        #else:
            #print('Parallel Elapsed time = ' + str(elapsed_time))
        return elapsed_time

    # Slave processes
    else:
        # 1. Receive data from Master process
        X_train_portion = comm.recv(source=masterID, tag=tag_x_train)
        y_train_portion = comm.recv(source=masterID, tag=tag_y_train)

        # 2. Find the remaining local sorted distances for the test data points
        kNN = KNN(k)
        kNN.fit(X_train_portion, y_train_portion)
        remaining_local_sorted_distances = kNN.getLocalSortedDistances(X_test)

        # 3. Send the remaining local sorted distances to the Master process
        comm.send(remaining_local_sorted_distances, dest=masterID, tag=tag_local_result)


#----------- Main -------------
A = 30
k = 5

if len(sys.argv) == 3:
	trainPath = sys.argv[1]
	testPath = sys.argv[2]
else:
	print("Paremetri sbagliati")
	sys.exit()

x_train, y_train = allocData(trainPath, A)
x_test, y_test = allocData(testPath, A)

executionTime = parallel_KNN(k, x_train, y_train, x_test, y_test)
print(executionTime)