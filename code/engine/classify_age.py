import numpy as np
import pandas as pd
from data_scripts.DataReader import *
from data_scripts.DataHolder import DataHolder
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from utils.config import get, is_file_prefix

def performance_CI(clf, X_test, y_test, loss_func):
    N = 1000
    bootstrap_performances = np.zeros(N)
    (n, d) = X_test.shape
    indices = np.arange(n)

    for i in range(N):
        sample_indices = np.random.choice(indices, size=n, replace=True)
        test_data = X_test[sample_indices]
        test_labels = y_test[sample_indices]

        y_pred = clf.decision_function(test_data)

        bootstrap_performances[i] = loss_func(test_labels, y_pred)

    bootstrap_performances = np.sort(bootstrap_performances)
    point_performance = np.mean(bootstrap_performances)

    return (point_performance, bootstrap_performances[25], bootstrap_performances[975])

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    dataHolder = DataHolder(readCSVData(get('DATA.PHENOTYPICS.PATH')))
    dataHolder.getMatricesFromPath(get('DATA.MATRICES.PATH'))
    X = np.array([mat[np.tril_indices(mat.shape[0], k=-1)] for mat in dataHolder.matrices])
    Y = dataHolder.getBinaryColumn('Sex', 'F', 'M')
    svmModel = SVC(kernel='rbf', class_weight='balanced')

    cValues = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    gammaValues = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    param_grid = {'C': cValues, 'gamma': gammaValues}
    folder = RepeatedKFold(n_splits=5, n_repeats=20)
    clf = GridSearchCV(clf,
                paramGrid, scoring='roc_auc', n_jobs=10,
                cv=folder, refit=True, verbose=1)

    numTestSplits = 10
    for i in range(numTestSplits):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        clf.fit(X_train, y_train)

        cvResults = pd.DataFrame.from_dict(clf.cv_results_)
        bestPerf = clf.best_score_
        bestCLF = clf.best_estimator_
        bestCLF.fit(X_train, y_train) #NOT SURE IF THIS IS NECESSARY
        bestC = clf.best_params_['C']
        bestGamma = clf.best_params_['gamma']
        (point, lower, upper) = performance_CI(bestCLF, X_test, y_test, LOSSFUNCTIONHERE)
        print('----------------------------------------------------------------')
        print('----------------------TEST SPLIT ' + str(i) + '-----------------------')
        print('----------------------------------------------------------------')
        cvResults['mean_test_score'] = -1.0 * cvResults['mean_test_score']
        cvResults['mean_train_score'] = -1.0 * cvResults['mean_train_score']
        print(cvResults[['param_C', 'param_gamma', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])
        print("Best C: " + str(bestC))
        print("Best Gamma: " + str(bestGamma))
        print("Performance on Test Set: " + '%f' % point + '(' + '%f' % lower + ',' + '%f' % upper + ')')
