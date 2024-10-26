import os
import sys
import numpy as np
import yaml

from sklearn.svm import SVR
from sklearn.decomposition import PCA

import utils

def transform_features_bonus(X):
    X = [list(x) for x in X]
    pca = PCA(n_components=50)
    pca.fit(X)
    X = pca.transform(X)
    return np.array(X)

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
    (X_test, y_test, qids_test, docids_test) = load_data_bonus(fold_dir)
    
    X_train, X_test = \
        transform_features_bonus(X_train), transform_features_bonus(X_test)
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
