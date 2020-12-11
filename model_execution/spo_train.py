# import sys

# sys.path.insert(0, '..')
# sys.path.insert(0, '../..')
# sys.path.insert(0, '.')

import os
# import os.path as osp
import numpy as np
import argparse
from pathlib import Path
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def combine_datasets(directory, data_full):
    try: 
        data_full = np.empty((0, num_features+1))
        for entry in os.scandir(directory):
            if entry.name.endswith('.csv'):
                # print(entry.path)
                data_full = np.append(data_full, pd.read_csv(entry.path, sep=',',header=None).values, axis=0)
    except OSError:
        if not os.path.exists(directory):
            raise

    return data_full

if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_train_dir", type=str)
    parser.add_argument("-data_test_dir", type=str, default='')
    parser.add_argument("-lp_train_dir", type=str, default='')
    parser.add_argument("-method", type=str, default='2stage')
    parser.add_argument("-timelimit", type=float, default=60)
    parser.add_argument("-logfile", type=str, default='sys.stdout')

    args = parser.parse_args()
    print(args)

    num_features = 11

    if args.method == '2stage':
        # aggregate data in args.data_train_dir
        data_train_full = np.empty((0, num_features+1))
        data_train_full = combine_datasets(args.data_train_dir, data_train_full)
        X_train, y_train = data_train_full[:,:num_features], data_train_full[:,-1]

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(X_train, y_train)

        # The coefficients
        print('Coefficients: \n', regr.coef_)

        if len(args.data_test_dir) > 0:
            data_test_full = np.empty((0, num_features+1))
            data_test_full = combine_datasets(args.data_test_dir, data_test_full)
            X_test, y_test = data_test_full[:,:num_features], data_test_full[:,-1]

            # Make predictions using the testing set
            y_pred = regr.predict(X_test)

            # The mean squared error
            print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

            # The coefficient of determination: 1 is perfect prediction
            print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))


    """ 
    evaluation
        
    1- read .lp file into cplex instane
    2- read .pk graph file
    3- iterate over variable nodes:
            c_i = model(features_i)
    4- store true objective then modify based on c_i predictions
    5- run mip solver
    6- evaluate solution on true objective
    7- write statistics about regret and solving time, etc.
     
    """
