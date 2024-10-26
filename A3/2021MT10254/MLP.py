import sys
import yaml

from sklearn.neural_network import MLPRegressor

import utils

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
    (X_val, y_val, qids_val, docids_val), \
    (X_test, y_test, qids_test, docids_test) = utils.load_data(fold_dir)
    
    X_train, X_val, X_test = \
        utils.transform_features(X_train), utils.transform_features(X_val), utils.transform_features(X_test)

    mlp_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, max_iter=max_iter, alpha=alpha, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, random_state=random_state)
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
