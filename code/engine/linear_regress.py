import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from utils.config import get, is_file_prefix

def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def performance_CI(regressor, X_test, y_test, loss_func):
    N = 1000
    bootstrap_performances = np.zeros(N)
    (n, d) = X_test.shape
    indices = np.arange(n)

    for i in range(N):
        sample_indices = np.random.choice(indices, size=n, replace=True)
        test_data = X_test[sample_indices]
        test_labels = y_test[sample_indices]

        y_pred = regressor.predict(test_data)

        bootstrap_performances[i] = loss_func(test_labels, y_pred)

    bootstrap_performances = np.sort(bootstrap_performances)
    point_performance = np.mean(bootstrap_performances)

    return (point_performance, bootstrap_performances[25], bootstrap_performances[975])

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    X = np.array([mat[np.tril_indices(mat.shape[0], k=-1)] for mat in dataHolder.matrices])
    Y = dataHolder._df['AgeYears'].values.copy()
    ridgeModel = Ridge(normalize=True)

    alphas = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001] #, 0.001, 0.01, 0.1, 1.0]
    solvers = ['cholesky']#, 'lsqr', 'sparse_cg', 'svd']
    param_grid = {'alpha': alphas, 'solver': solvers}
    folder = RepeatedKFold(n_splits=5, n_repeats=20)
    regressor = GridSearchCV(ridgeModel,
                param_grid, scoring='neg_mean_squared_error', n_jobs=10,
                cv=folder, refit=True, verbose=1)
    numTestSplits = 10
    for i in range(numTestSplits):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        regressor.fit(X_train, y_train)

        cvResults = pd.DataFrame.from_dict(regressor.cv_results_)
        bestPerf = regressor.best_score_
        bestRidge = regressor.best_estimator_
        bestRidge.fit(X_train, y_train) #NOT SURE IF THIS IS NECESSARY
        bestAlpha = regressor.best_params_['alpha']
        bestSolver = regressor.best_params_['solver']
        (point, lower, upper) = performance_CI(bestRidge, X_test, y_test, MSE)
        print('----------------------------------------------------------------')
        print('----------------------TEST SPLIT ' + str(i) + '-----------------------')
        print('----------------------------------------------------------------')
        cvResults['mean_test_score'] = -1.0 * cvResults['mean_test_score']
        cvResults['mean_train_score'] = -1.0 * cvResults['mean_train_score']
        print(cvResults[['param_alpha', 'param_solver', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])
        print("Best Alpha: " + str(bestAlpha))
        print("Best Solver: " + str(bestSolver))
        print("Performance on Test Set: " + '%f' % point + '(' + '%f' % lower + ',' + '%f' % upper + ')')
