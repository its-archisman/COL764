import os
import sys
import numpy as np
from time import time
import yaml

from sklearn.ensemble import GradientBoostingRegressor

import utils

def transform_features_bonus(X):
    X = [list(x) for x in X]
    for i in range(len(X)):
        val1 = 0
        if X[i][0] > 0:
            val1 = np.log(X[i][0])
        X[i].append(val1)
    X = np.array(X)
    return X


def load_data_bonus(fold_directory):
    train_file = os.path.join(fold_directory, 'bonus.train.txt')
    val_file = os.path.join(fold_directory, 'bonus.test.txt')
    def load_file(file_path, val=0):
        data, labels, qids, docids = [], [], [], []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                qid = parts[1].split(':')[1]
                label = float(parts[0])
                features = [float(x.split(':')[1]) for x in parts[2:]][:60]
                data.append(features)
                labels.append(label)
                qids.append(qid)
        if val:
            docid_dict = {}
            count = 1
            for i in range(len(labels)):
                key = " ".join(str(ele) for ele in data[i])
                if key not in docid_dict:
                    docid_dict[key] = str(count)
                    count += 1
                docids.append(docid_dict[key])
        return np.array(data), np.array(labels), qids, docids
    
    X_train, y_train, qids_train, docids_train = load_file(train_file)
    X_val, y_val, qids_val, docids_val = load_file(val_file, 1)

    return (X_train, y_train, qids_train, []), (X_val, y_val, qids_val, docids_val)


def process_yaml_GBDT(param_file_path):
    with open(param_file_path, 'r') as file:
        best_hyperparameters = yaml.safe_load(file)
    conversions = {
        'subsample': float,
        'n_estimators': int,
        'learning_rate': float,
        'max_depth': int,
        'criterion': str,
        'random_state': int,
        'max_features': str,
        'alpha': float,
    }
    for key, func in conversions.items():
        best_hyperparameters[key] = func(best_hyperparameters[key])
    return best_hyperparameters


def do_GBDT(fold_dir, output_file_path, subsample, n_estimators, learning_rate, max_depth, criterion, random_state, max_features, alpha,):
    (X_train, y_train, qids_train, docids_train), \
    (X_test, y_test, qids_test, docids_test) = load_data_bonus(fold_dir)
    
    X_train, X_test = \
        transform_features_bonus(X_train), transform_features_bonus(X_test)
    gbdt_model = GradientBoostingRegressor(
        subsample=subsample, 
        n_estimators=n_estimators, 
        learning_rate=learning_rate, 
        max_depth=max_depth, 
        criterion=criterion, 
        random_state=random_state, 
        max_features=max_features, 
        alpha=alpha,
    )
    gbdt_model = utils.train_model(gbdt_model, X_train, y_train)
    gbdt_model = utils.train_model(gbdt_model, X_train, y_train)
    
    testing_results = utils.predict_model(gbdt_model, X_test, qids_test, docids_test)    
    
    utils.write_results(output_file_path, testing_results)
    
    
def main():
    fold_dir = sys.argv[1]
    output_file_path = sys.argv[2]
    param_file_path = sys.argv[3]

    params = process_yaml_GBDT(param_file_path)
    do_GBDT(fold_dir, output_file_path, **params)

if __name__ == '__main__':
    main()
