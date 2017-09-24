import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from utils.config import get, is_file_prefix

def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def crossValidate(ridge_model, X, Y, k=5, numReps=10):
    validationPerformance = []

    for i in range(numReps):
        skf = KFold(n_splits=k)
        for trainIndex, valdIndex in skf.split(X, Y):
            X_train, X_vald = X[trainIndex], X[valdIndex]
            y_train, y_vald = Y[trainIndex], Y[valdIndex]

            ridge_model.fit(X_train, y_train)
            y_pred = ridge_model.predict(X_vald)
            error = RMSE(y_pred, y_vald)
            print('CV error: ' + '%f' % error)
            validationPerformance.append(error)

    return np.mean(validationPerformance)

def getTestPerformance(X, Y, alphas):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    averageCVPerf = []
    testPerf = []
    for alpha in alphas:
        print('--------------------------------------------')
        print('--------- alpha: ' + '%f' % alpha +  '----------')
        print('--------------------------------------------')
        ridge_model = Ridge(alpha=alpha, normalize=True)
        averageCVPerf.append(crossValidate(ridge_model, X, Y, k=5, numReps=20))

        ridge_model.fit(X_train, y_train)
        y_pred = ridge_model.predict(X_test)
        error = RMSE(y_pred, y_test)
        print('TEST error: ' + '%f' % error)
        testPerf.append(error)
    return (np.array(averageCVPerf), np.array(testPerf))

def runTests(X, Y, alphas, numReps=10):
    summedCVPerf = np.zeros(len(alphas))
    summedTestPerf = np.zeros(len(alphas))
    for i in range(numReps):
        (averageCVPerf, testPerf) = getTestPerformance(X, Y, alphas)
        summedCVPerf += averageCVPerf
        summedTestPerf += testPerf

    averageCVPerf = summedCVPerf / numReps
    averageTestPerf = summedTestPerf / numReps
    return (averageCVPerf, averageTestPerf)

def performance_CI(clf, X, y):
    N = 1000
    bootstrap_performances = np.zeros(N)
    (n, d) = X.shape
    indices = np.arange(n)

    for i in range(N):
        sample_indices = np.random.choice(indices, size=n, replace=True)
        test_data = X[sample_indices]
        test_labels = y[sample_indices]

        y_pred = clf.predict(test_data)

        bootstrap_performances[i] = RMSE(test_labels, y_pred)

    bootstrap_performances = np.sort(bootstrap_performances)
    point_performance = np.mean(bootstrap_performances)

    return (point_performance, bootstrap_performances[25], bootstrap_performances[975])


if __name__ == '__main__':
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    X = np.array([mat[np.tril_indices(mat.shape[0], k=-1)] for mat in dataHolder.matrices])
    Y = dataHolder._df['AgeYears'].values.copy()
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    (averageCVPerf, averageTestPerf) = runTests(X, Y, alphas, numReps=100)
    print('--------------------------------------------')
    print('---------------FINAL RESULTS----------------')
    print('--------------------------------------------')
    minIndex = 0
    minPerf = averageCVPerf[minIndex]
    for i in range(len(alphas)):
        print('Alpha: ' + '%f' % alphas[i] + ' CV: ' + '%f' % averageCVPerf[i] + ' TEST: ' '%f' % averageTestPerf[i])
        if (averageCVPerf[i] < minPerf):
            minPerf = averageCVPerf[i]
            minIndex = i

    clf = Ridge(alpha=alphas[minIndex], normalize=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf.fit(X_train, y_train)
    (p, l, u) = performance_CI(clf, X_test, Y_test)
    print("Confidence Interval Perf: " + str(p) + " (" + str(l) + "," + str(u) + ")")
