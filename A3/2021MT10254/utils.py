import os
from time import time

import numpy as np

def load_data(fold_directory):
    train_file = os.path.join(fold_directory, 'trainingset.txt')
    val_file = os.path.join(fold_directory, 'validationset.txt')
    test_file = os.path.join(fold_directory, 'testset.txt')

    def load_file(file_path):
        data, labels, qids, docids = [], [], [], []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                qid = parts[1].split(':')[1]
                docid = parts[-1]
                label = float(parts[0])
                features = [float(x.split(':')[1]) for x in parts[2:-3]]
                data.append(features)
                labels.append(label)
                qids.append(qid)
                docids.append(docid)

        return np.array(data), np.array(labels), qids, docids

    X_train, y_train, qids_train, docids_train = load_file(train_file)
    X_val, y_val, qids_val, docids_val = load_file(val_file)
    X_test, y_test, qids_test, docids_test = load_file(test_file)

    return (X_train, y_train, qids_train, docids_train), (X_val, y_val, qids_val, docids_val), (X_test, y_test, qids_test, docids_test)

def train_model_and_get_results(regressor, X_train, y_train, X_test, qids_test, docids_test):
    start_time = time()
    regressor.fit(X_train, y_train)
    training_time = time() - start_time

    start_time = time()
    predictions = regressor.predict(X_test)
    test_time = time() - start_time

    results = format_results(predictions, qids_test, docids_test)
    return results, training_time, test_time

def train_model(regressor, X_train, y_train):
    start_time = time()
    regressor.fit(X_train, y_train)
    training_time = time() - start_time
    return regressor

def predict_model(regressor, X_test, qids_test, docids_test):
    start_time = time()
    predictions = regressor.predict(X_test)
    test_time = time() - start_time

    results = format_results(predictions, qids_test, docids_test)
    return results

def format_results(predictions, qids_test, docids_test):
    results = {}
    for i in range(len(qids_test)):
        qid = qids_test[i]
        if qid not in results:
            results[qid] = []
        results[qid].append([docids_test[i], predictions[i]])
    
    for key in results:
        results[key].sort(key=lambda x: x[1], reverse=True)
    return results

def write_results(output_file, results):
    f = open(output_file, 'w')
    f.write("qid\titeration\tdocid\trelevancy\n")
    for qid in sorted(results.keys()):
        for tup in results[qid]:
            f.write(f'{qid}\t0\t{tup[0]}\t{tup[1]}\n')


def transform_features(X):
    X = [list(x) for x in X]
    for i in range(len(X)):
        val1 = 0
        if X[i][0] > 0:
            val1 = np.log(X[i][0])
        X[i].append(val1)
    X = np.array(X)
    return X
