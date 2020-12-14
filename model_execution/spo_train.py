import os
import numpy as np
import argparse
from pathlib import Path
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn import neural_network
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import pandas as pd
import pickle
import time


def combine_datasets(directory, dim):
    try: 
        data_full = np.empty((0, dim))
        for entry in os.scandir(directory):
            if entry.name.endswith('.csv'):
                # print(entry.path)
                data_full = np.append(data_full, pd.read_csv(entry.path, sep=',',header=None).values, axis=0)
                print(data_full.shape)
    except OSError:
        if not os.path.exists(directory):
            raise

    return data_full

def test_model(model, X, y):
    # Make predictions using the testing set
    time_test = time.time()
    y_pred = model.predict(X)
    time_test = time.time() - time_test

    # The mean squared error
    mse_test = mean_squared_error(y, y_pred)
    print('Mean squared error: %.2f' % mse_test)

    # The coefficient of determination: 1 is perfect prediction
    r2_test = r2_score(y, y_pred)
    print('Coefficient of determination: %.2f' % r2_test)

    return time_test, mse_test, r2_test


if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_train_dir", type=str)
    parser.add_argument("-data_test_dir", type=str, default='')
    parser.add_argument("-lp_train_dir", type=str, default='')
    parser.add_argument("-method", type=str, default='2stage')
    parser.add_argument("-timelimit", type=float, default=60)

    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-single_model", type=bool, default=False)
    parser.add_argument("-model_type", type=str, default='linear')
    parser.add_argument("-poly_degree", type=int, default=2)

    args = parser.parse_args()
    print(args)

    # output directories
    output_dir = "SPO_MODELS/" + args.output_dir
    try: 
        os.makedirs(output_dir)
    except OSError:
        if not os.path.exists(output_dir):
            raise

    num_features = 10
    dim = num_features + 2

    time_dataread, time_train, time_test = 0.0, 0.0, 0.0
    mse_train, r2_train = 0.0, 0.0
    mse_test, r2_test = 0.0, 0.0

    if args.method == '2stage':
        # aggregate data in args.data_train_dir
        time_dataread = time.time()
        data_train_full = combine_datasets(args.data_train_dir, dim=dim)

        if args.single_model:
            data_train_full[:,0] = 0

        if len(args.data_test_dir) > 0:
            data_test_full = combine_datasets(args.data_test_dir, dim=dim)

            if args.single_model:
                data_test_full[:,0] = 0

        model_indicators = np.unique(data_train_full[:,0])
        time_dataread = time.time() - time_dataread

        for indicator in model_indicators:
            rows_indicator = np.where(data_train_full[:,0] == indicator)[0]
            X_train, y_train = data_train_full[rows_indicator,1:num_features+1], data_train_full[rows_indicator,-1]

            # Create linear regression object
            if args.model_type == 'linear':
                regr = linear_model.LinearRegression()
            elif args.model_type == 'svm_poly':
                regr = svm.SVR(kernel='poly', degree=args.poly_degree)
            elif args.model_type == 'poly_ridge':
                regr = make_pipeline(PolynomialFeatures(args.poly_degree), Ridge())
            elif args.model_type == 'mlp':
                regr = neural_network.MLPRegressor((10,10))

            # Train the model using the training sets
            time_train = time.time()
            regr.fit(X_train, y_train)
            time_train = time.time() - time_train

            _, mse_train, r2_train = test_model(regr, X_train, y_train)

            # Write model to file
            model_filename = '%s/%s_%d_%d.pk' % (output_dir, args.model_type, args.poly_degree, indicator)
            pickle.dump(regr, open(model_filename, 'wb'))

            if len(args.data_test_dir) > 0:
                rows_indicator = np.where(data_test_full[:,0] == indicator)[0]
                X_test, y_test = data_test_full[rows_indicator,1:num_features+1], data_test_full[rows_indicator,-1]

                time_test, mse_test, r2_test = test_model(regr, X_test, y_test)

            results_str = "%d,%s,%g,%g,%g,%g,%g,%g,%g,%s" % (
                indicator,
                args.model_type,
                time_train,
                mse_train, 
                r2_train,
                time_test,
                mse_test, 
                r2_test,
                time_dataread,
                model_filename)

            print(results_str)

            results_filename = '%s/%s_%d_%d.csv' % (output_dir, args.model_type, args.poly_degree, indicator)
            with open(results_filename, "w+") as results_file:
                results_file.write(results_str)

