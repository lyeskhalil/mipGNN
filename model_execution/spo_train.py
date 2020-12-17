import os
import sys
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
import cplex

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from spo_torch import SPONet, SPOLoss
import spo_utils


def combine_datasets(directory, dim, operation='combine', poly_degree=1):
    try:
        if operation == 'combine': 
            data_full = np.empty((0, dim))
        elif operation == 'list':
            data_full = []

        if poly_degree > 1:
            poly = PolynomialFeatures(poly_degree, include_bias=False)

        filenames = []
        for entry in os.scandir(directory):
            if entry.name.endswith('.csv'):
                data_cur = pd.read_csv(entry.path, sep=',',header=None).values
                
                if poly_degree > 1:
                    features = poly.fit_transform(data_cur[:,1:-1])
                    data_cur = np.concatenate((
                        data_cur[:,0:1], 
                        features, 
                        np.expand_dims(data_cur[:,-1], axis=1)), axis=1)

                if operation == 'combine': 
                    data_full = np.append(data_full, data_cur, axis=0)
                
                elif operation == 'list':
                    data_full += [torch.tensor(data_cur)]
                    print(data_full[-1].size())
                
                filenames += [entry.path[entry.path.find('/SPO_DATA/')+10:-4]]
    except OSError:
        if not os.path.exists(directory):
            raise

    return data_full, filenames

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
    parser.add_argument("-method", type=str, default='2stage')
    parser.add_argument("-data_train_dir", type=str)
    parser.add_argument("-data_test_dir", type=str, default='')

    parser.add_argument("-output_dir", type=str)
    parser.add_argument("-single_model", type=int, default=0)
    parser.add_argument("-model_type", type=str, default='linear')
    parser.add_argument("-poly_degree", type=int, default=2)
    parser.add_argument("-ridge_reg", type=float, default=1.0)

    # Training parameters
    parser.add_argument("-nn_epochs", type=int, default=1000)
    parser.add_argument("-nn_lr_init", type=float, default=1e-5)
    parser.add_argument("-nn_lr_decay", type=int, default=1)
    parser.add_argument("-nn_reg", type=float, default=1e-5)
    parser.add_argument("-nn_termination", type=float, default=0.05)
    parser.add_argument("-nn_patience", type=int, default=20)

    # Tensorboard parameters
    parser.add_argument("-nn_tb_dir", type=str, default='SPO_TENSORBOARD')

    # Architecture parameters
    parser.add_argument("-nn_depth", type=int, default=0)
    parser.add_argument("-nn_width", type=int, default=100)

    # CPLEX parameters
    parser.add_argument("-nn_cpx_timelimit", type=float, default=60)
    parser.add_argument("-nn_cpx_threads", type=int, default=4)

    # Warmstart parameters
    parser.add_argument("-nn_warmstart_dir", type=str, default='')
    parser.add_argument("-nn_warmstart_prefix", type=str, default='')

    # Data parameters
    parser.add_argument("-nn_poly_degree", type=int, default=1)

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
        data_train_full, _ = combine_datasets(args.data_train_dir, dim=dim)

        if args.single_model:
            data_train_full[:,0] = 0

        if len(args.data_test_dir) > 0:
            data_test_full, _ = combine_datasets(args.data_test_dir, dim=dim)

            if args.single_model:
                data_test_full[:,0] = 0

        model_indicators = np.unique(data_train_full[:,0])
        time_dataread = time.time() - time_dataread

        for indicator in model_indicators:
            rows_indicator = np.where(data_train_full[:,0] == indicator)[0]
            X_train, y_train = data_train_full[rows_indicator,1:num_features+1], data_train_full[rows_indicator,-1]

            # Create linear regression object
            model_filename = args.model_type
            if args.model_type == 'linear':
                regr = linear_model.LinearRegression()
            elif args.model_type == 'svm_poly':
                regr = svm.SVR(kernel='poly', degree=args.poly_degree)
                model_filename += '_' + args.poly_degree
            elif args.model_type == 'ridge_poly':
                regr = make_pipeline(PolynomialFeatures(args.poly_degree), Ridge(alpha=args.ridge_reg))
                model_filename += '_%d_%g' % (args.poly_degree, args.ridge_reg)
            elif args.model_type == 'mlp':
                regr = neural_network.MLPRegressor((10,10))

            # Train the model using the training sets
            time_train = time.time()
            regr.fit(X_train, y_train)
            time_train = time.time() - time_train

            _, mse_train, r2_train = test_model(regr, X_train, y_train)

            # Write model to file
            model_filename = '%s/%s_%d.pk' % (output_dir, model_filename, indicator)
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

    elif args.method == 'spo':
        torch.manual_seed(0)
        warmstart_bool = int(args.nn_warmstart_dir != '' and args.nn_warmstart_prefix != '')

        filename_noext = 'depth_%d_width_%d_reg_%g_polydeg_%d_lrinit_%g_lrdecay_%d_warmst_%d' % (
            args.nn_depth, 
            args.nn_width,
            args.nn_reg,
            args.nn_poly_degree, 
            args.nn_lr_init,
            args.nn_lr_decay,
            warmstart_bool)
        model_filename = '%s/%s.pt' % (output_dir, filename_noext)
        tb_dirname = '%s/%s' % (args.nn_tb_dir, filename_noext)

        # Tensorboard setup
        tb_writer = SummaryWriter(tb_dirname)

        num_epochs = args.nn_epochs
        model_indicators = [0,1]

        dtype = torch.double
        device = torch.device("cpu")
        torch.set_default_dtype(dtype)

        data_train_full, data_filenames = combine_datasets(args.data_train_dir, dim=dim, operation='list', poly_degree=args.nn_poly_degree)

        num_features = len(data_train_full[0][0,:]) - 2

        sol_true = []
        objval_true = []
        instance_cpx = []
        coeffs_true = []
        model_indices = []

        to_delete_data = []
        for instance_idx, instance in enumerate(data_filenames):
            file_npz = "../gisp_generator/SOL/%s.npz" % instance
            file_lp = "../gisp_generator/LP/%s.lp" % instance
            if not os.path.isfile(file_npz) or not os.path.isfile(file_lp):
                to_delete_data += [instance_idx]
                continue
            # get true opt
            sol_pool = np.load(file_npz)['solutions']
            sol_true += [torch.unsqueeze(torch.tensor(sol_pool[0, 1:]), 1)]
            objval_true += [torch.tensor([sol_pool[0,0]])]
            
            instance_cpx += [cplex.Cplex(file_lp)]

            coeffs_true += [torch.unsqueeze(torch.tensor(instance_cpx[-1].objective.get_linear()), 1)]

            model_indices += [[data_train_full[instance_idx][:,0] == 0, data_train_full[instance_idx][:,0] == 1]]

            if instance_cpx[-1].objective.sense[instance_cpx[-1].objective.get_sense()] == 'maximize':
                instance_cpx[-1].objective.set_sense(instance_cpx[-1].objective.sense.minimize)
                coeffs_true[-1] *= -1
                objval_true[-1] *= -1

            spo_utils.disable_output_cpx(instance_cpx[-1])
            instance_cpx[-1].parameters.timelimit.set(args.nn_cpx_timelimit)
            instance_cpx[-1].parameters.emphasis.mip.set(1)
            instance_cpx[-1].parameters.threads.set(args.nn_cpx_threads)
        
        for counter, idx in enumerate(to_delete_data):
            del data_train_full[idx - counter]

        num_instances = len(sol_true)

        loss_fn = SPOLoss.apply
        depth, width = args.nn_depth, args.nn_width
        models = [SPONet(num_features, depth, width, relu_sign=-1), SPONet(num_features, depth, width, relu_sign=1)]

        if args.nn_warmstart_dir != '' and args.nn_warmstart_prefix != '' and depth == 0:
            models_pretrained = spo_utils.read_sklearn_model(args.nn_warmstart_dir, args.nn_warmstart_prefix)
            assert(len(models_pretrained) == len(model_indicators))
            for indicator in model_indicators:
                models[indicator].layers[0].weight.data = torch.unsqueeze(torch.tensor(models_pretrained[indicator].coef_), 0)
                models[indicator].layers[0].bias.data = torch.unsqueeze(torch.tensor(models_pretrained[indicator].intercept_), 0)

        optimizer = optim.SGD(
            list(models[0].parameters())+list(models[1].parameters()), 
            lr=args.nn_lr_init)

        lmbda = lambda epoch: args.nn_lr_init/(np.sqrt(epoch+1))
        # lmbda = lambda epoch: 2.0*1e-6/(args.nn_reg * (epoch+2))
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
        
        running_loss_best, running_loss_withreg_best, epoch_best = np.inf, np.inf, -1
        time_solve = 0.0
        for epoch in range(num_epochs):
            print('---------------')
            if args.nn_lr_decay:
                optimizer.param_groups[0]['lr'] = lmbda(epoch)

            running_loss, running_loss_best = 0.0, np.inf
            running_loss_withreg, running_loss_withreg_best = 0.0, np.inf
            for instance_idx in range(num_instances): #len(data_train_full)
                sys.stdout.flush()
                optimizer.zero_grad()

                indices = model_indices[instance_idx]

                coeffs = [models[0](data_train_full[instance_idx][indices[0], 1:num_features+1]), models[1](data_train_full[instance_idx][indices[1], 1:num_features+1])]

                coeffs = torch.cat((coeffs[0], coeffs[1]), 0)

                # todo: track CPLEX solution statuses
                time_cur = time.time()
                loss_val = loss_fn(coeffs, sol_true[instance_idx], coeffs_true[instance_idx], instance_cpx[instance_idx])
                time_solve += time.time() - time_cur

                loss_spo = float(loss_val - objval_true[instance_idx])

                loss_val += args.nn_reg * torch.norm(coeffs) 

                loss_val.backward()

                optimizer.step()

                running_loss += loss_spo / (1e-9 + np.abs(objval_true[instance_idx])) * num_instances
                running_loss_withreg += loss_val.data

                print("SPO loss [%d-%d] = %g = (%g) - (%g)" % (epoch, instance_idx, loss_spo, loss_val, objval_true[instance_idx]))
                print("Learning rate =", optimizer.param_groups[0]['lr'])#scheduler.get_lr()[0])

            if running_loss < running_loss_best or running_loss_withreg < running_loss_withreg_best:
                running_loss_best = running_loss
                running_loss_withreg_best = running_loss_withreg
                epoch_best = epoch

                torch.save({
                    'epoch': epoch, 
                    'num_features': num_features,
                    'nn_poly_degree': args.nn_poly_degree,
                    'model0_state_dict': models[0].state_dict(),
                    'model1_state_dict': models[1].state_dict(),
                    'loss_spo': running_loss_best,
                    'loss_spo_withreg': running_loss_withreg_best
                    }, model_filename)

                # model0_loaded = SPONet(num_features, depth, width, relu_sign=-1)
                # checkpoint = torch.load(model_filename)
                # model0_loaded.load_state_dict(checkpoint['model0_state_dict'])

            tb_writer.add_scalar('training loss', running_loss, epoch)
            tb_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
            tb_writer.add_scalar('solver time', time_solve, epoch)
            tb_writer.add_histogram('model0 gradients', models[0].layers[0].weight.grad, epoch)
            tb_writer.add_histogram('model1 gradients', models[1].layers[0].weight.grad, epoch)
            tb_writer.add_histogram('model0 weights', models[0].layers[0].weight.data, epoch)
            tb_writer.add_histogram('model1 weights', models[1].layers[0].weight.data, epoch)

            if epoch - epoch_best >= args.nn_patience or running_loss_best <= args.nn_termination:
                print("Early termination!")
                print("epoch - epoch_best =", epoch - epoch_best)
                print("running_loss_best =", running_loss_best)

                break