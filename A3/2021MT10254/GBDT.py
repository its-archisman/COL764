import os
import sys
import yaml

from sklearn.ensemble import GradientBoostingRegressor

import utils

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
    (X_val, y_val, qids_val, docids_val), \
    (X_test, y_test, qids_test, docids_test) = utils.load_data(fold_dir)
    
    X_train, X_val, X_test = \
        utils.transform_features(X_train), utils.transform_features(X_val), utils.transform_features(X_test)

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
