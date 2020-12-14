import os
# import os.path as osp
import numpy as np
import argparse
from pathlib import Path
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import pandas as pd
import pickle


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

if __name__ == '__main__':

    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_train_dir", type=str)
    parser.add_argument("-data_test_dir", type=str, default='')
    parser.add_argument("-lp_train_dir", type=str, default='')
    parser.add_argument("-method", type=str, default='2stage')
    parser.add_argument("-single_model", type=bool, default=False)
    parser.add_argument("-model_type", type=str, default='linear')
    parser.add_argument("-model_path", type=str)
    parser.add_argument("-timelimit", type=float, default=60)
    parser.add_argument("-logfile", type=str, default='sys.stdout')

    args = parser.parse_args()
    print(args)

    num_features = 10
    dim = num_features + 2

    if args.method == '2stage':
        # aggregate data in args.data_train_dir
        data_train_full = combine_datasets(args.data_train_dir, dim=dim)

        if args.single_model:
            data_train_full[:,0] = 0

        if len(args.data_test_dir) > 0:
            data_test_full = combine_datasets(args.data_test_dir, dim=dim)

            if args.single_model:
                data_test_full[:,0] = 0

        model_indicators = np.unique(data_train_full[:,0])

        for indicator in model_indicators:
            rows_indicator = np.where(data_train_full[:,0] == indicator)[0]
            X_train, y_train = data_train_full[rows_indicator,1:num_features+1], data_train_full[rows_indicator,-1]

            # Create linear regression object
            regr = linear_model.LinearRegression()
            # regr = SVR(kernel='poly', degree=2)

            # Train the model using the training sets
            regr.fit(X_train, y_train)

            # The coefficients
            # print('Coefficients: \n', regr.coef_)

            # Write model to file
            model_filename = '%s_%d.pk' % (args.model_path, indicator)
            pickle.dump(regr, open(model_filename, 'wb'))
            # clf2 = pickle.load(open(model_filename, 'rb'))
            # np.savetxt('%s_%d.csv' % (args.model_path, indicator), regr.coef_, delimiter=',')

            if len(args.data_test_dir) > 0:
                rows_indicator = np.where(data_test_full[:,0] == indicator)[0]
                X_test, y_test = data_test_full[rows_indicator,1:num_features+1], data_test_full[rows_indicator,-1]

                # Make predictions using the testing set
                y_pred = regr.predict(X_test)

                # The mean squared error
                print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

                # The coefficient of determination: 1 is perfect prediction
                print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
