import os
import sys
import numpy as np
import yaml

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

import utils

def process_yaml_SVR(param_file_path):
    with open(param_file_path, 'r') as file:
        best_hyperparameters = yaml.safe_load(file)
    conversions = {
        'kernel': str,
        'C': float,
        'epsilon': float,
        'degree': int,
        'tol': float,
        'max_iter': int,
        'shrinking': bool,
        'gamma': str
    }
    for key, func in conversions.items():
        best_hyperparameters[key] = func(best_hyperparameters[key])
    return best_hyperparameters


def do_SVR(fold_dir, output_file_path, kernel, C, epsilon, degree, tol, max_iter, shrinking, gamma):
    (X_train, y_train, qids_train, docids_train), \
    (X_val, y_val, qids_val, docids_val), \
    (X_test, y_test, qids_test, docids_test) = utils.load_data(fold_dir)
    
    scaler = MinMaxScaler()
    X_train, X_val = scaler.fit_transform(X_train), scaler.fit_transform(X_val)
    

    X_train, X_val, X_test = \
        utils.transform_features(X_train), utils.transform_features(X_val), utils.transform_features(X_test)

    svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon, degree=degree, tol=tol, max_iter=max_iter, shrinking=shrinking, gamma=gamma)
    svr_model = utils.train_model(svr_model, X_train, y_train)

    testing_results = utils.predict_model(svr_model, X_test, qids_test, docids_test)    
    utils.write_results(output_file_path, testing_results)
    
    
def main():
    fold_dir = sys.argv[1]
    output_file_path = sys.argv[2]
    param_file_path = sys.argv[3]

    params = process_yaml_SVR(param_file_path)
    do_SVR(fold_dir, output_file_path, **params)

if __name__ == '__main__':
    main()
