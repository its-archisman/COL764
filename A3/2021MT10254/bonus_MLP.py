import os
import sys
import numpy as np
from time import time
import yaml

from sklearn.neural_network import MLPRegressor
import utils

def transform_features_bonus(X):
    X = [list(x) for x in X]

    for i in range(len(X)):
        val1 = 0
        if X[i][0] > 0:
            val1 = np.log(X[i][0])
        X[i].append(val1)
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

def process_yaml_MLP(param_file_path):
    with open(param_file_path, 'r') as file:
        best_hyperparameters = yaml.safe_load(file)
    conversions = {
        'hidden_layer_sizes': None,
        'activation': str,
        'solver': str,
        'max_iter': int,
        'alpha': float,
        'learning_rate': str,
        'learning_rate_init': float,
        'power_t': float,
        'random_state': int,
    }
    for key, func in conversions.items():
        if key == 'hidden_layer_sizes':
            s = best_hyperparameters[key].split(',')
            best_hyperparameters[key] = (int(s[0][1:]),)
            continue
        if key == 'random_state' and best_hyperparameters[key] == 'None':
            best_hyperparameters[key] = None
            continue
        best_hyperparameters[key] = func(best_hyperparameters[key])
    return best_hyperparameters


def do_MLP(fold_dir, output_file_path, hidden_layer_sizes, activation, solver, max_iter, alpha, learning_rate, learning_rate_init, power_t, random_state):

    (X_train, y_train, qids_train, docids_train), \
    (X_test, y_test, qids_test, docids_test) = load_data_bonus(fold_dir)

    mlp_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter, alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, random_state=random_state)
    mlp_model = utils.train_model(mlp_model, X_train, y_train)

    mlp_model = utils.train_model(mlp_model, X_train, y_train)
    testing_results = utils.predict_model(mlp_model, X_test, qids_test, docids_test)
    utils.write_results(output_file_path, testing_results)
    
    
def main():
    fold_dir = sys.argv[1]
    output_file_path = sys.argv[2]
    param_file_path = sys.argv[3]

    params = process_yaml_MLP(param_file_path)
    do_MLP(fold_dir, output_file_path, **params)

if __name__ == '__main__':
    main()
