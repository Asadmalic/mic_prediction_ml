
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score


# a function for preparing our training and testing data
def prep_data(phenotype):
    pheno = pd.read_csv('./data/metadata.csv', index_col=0)
    pheno = pheno.dropna(subset=[phenotype])  # drop samples that don't have a value for our chosen resistance profile
    pheno = pheno[phenotype]

    # read in unitig data
    X = pd.read_csv('./data/' + phenotype + '_unitigs.Rtab', sep=" ", index_col=0,
                    low_memory=False)
    X = X.transpose()
    X = X[X.index.isin(pheno.index)]  # only keep rows with a resistance measure
    pheno = pheno[pheno.index.isin(X.index)]
    return X, pheno


# function for fitting a model
def fitmodel(X, pheno, estimator, parameters, modelname, method, performance, times):
    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(X, pheno):
        # time how long it takes to train each model type
        start = time.process_time()

        # split data into train/test sets
        X_train = X.iloc[train_index]
        y_train = pheno[train_index]
        X_test = X.iloc[test_index]
        y_test = pheno[test_index]

        # perform grid search to identify best hyper-parameters
        gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3, n_jobs=-1, scoring='balanced_accuracy')
        gs_clf.fit(X_train, y_train)

        # predict resistance in test set
        y_pred = gs_clf.predict(X_test)
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred > 0.5] = 1

        score = balanced_accuracy_score(y_test, y_pred)
        performance = np.append(performance, score)
        method = np.append(method, modelname)
        times = np.append(times, (time.process_time() - start))

        print("Best hyperparameters for this fold")
        print(gs_clf.best_params_)
        print("Confusion matrix for this fold")
        print(confusion_matrix(y_test, y_pred))
    return gs_clf, method, performance, times


def get_x_y(target_name:str, return_dataframe=True, threshold=200):

    target = pd.read_csv('./data/metadata.csv', index_col=0)
    target = target.dropna(subset=[f'{target_name}_mic'])  # drop samples that don't have a value for our chosen resistance profile
    target = target[f'{target_name}_mic']

    target = target[target < threshold]

    # read in unitig data
    X = pd.read_csv('./data/' + f'{target_name}_sr' + '_unitigs.Rtab',
                    sep=" ", index_col=0,
                    low_memory=False)

    X = X.transpose()
    X = X[X.index.isin(target.index)]  # only keep rows with a resistance measure

    target = target[target.index.isin(X.index)]

    df = pd.DataFrame(
        np.concatenate([X.values, target.values.reshape(-1, 1)], axis=1),
        columns=list(X.columns) + ['output'])

    if return_dataframe:
        return df

    return X, target
