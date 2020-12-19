# todo: check CPLEX status
# todo: solve LP instead of MIP

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
import multiprocess as mp

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import spo_torch
from spo_torch import SPONet, SPOLoss
import spo_utils


def build_dict(data_filenames, data_full, args):
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

        model_indices += [[data_full[instance_idx][:,0] == 0, data_full[instance_idx][:,0] == 1]]

        if instance_cpx[-1].objective.sense[instance_cpx[-1].objective.get_sense()] == 'maximize':
            instance_cpx[-1].objective.set_sense(instance_cpx[-1].objective.sense.minimize)
            coeffs_true[-1] *= -1
            objval_true[-1] *= -1

        spo_utils.disable_output_cpx(instance_cpx[-1])
        instance_cpx[-1].parameters.timelimit.set(args.nn_cpx_timelimit)
        instance_cpx[-1].parameters.emphasis.mip.set(1)
        instance_cpx[-1].parameters.threads.set(args.nn_cpx_threads)

    dict_all = {}
    dict_all['sol_true'] = sol_true
    dict_all['objval_true'] = objval_true
    dict_all['model_indices'] = model_indices
    dict_all['instance_cpx'] = instance_cpx
    dict_all['coeffs_true'] = coeffs_true

    for counter, idx in enumerate(to_delete_data):
        del data_full[idx - counter]

    dict_all['data'] = data_full
    dict_all['num_instances'] = len(sol_true)

    return dict_all


def combine_datasets(directory, operation='combine', poly_degree=1):
    try:
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
                    if len(filenames) == 0:
                        data_full = np.empty((0, data_cur.shape[1]))
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


def main(args):
    """ Parse arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("-method", type=str, default='2stage')
    parser.add_argument("-data_train_dir", type=str)
    parser.add_argument("-data_validation_dir", type=str, default='')

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
    parser.add_argument("-nn_patience", type=int, default=200)
    parser.add_argument("-nn_batchsize", type=int, default=1)
    parser.add_argument("-nn_poolsize", type=int, default=1)

    # Tensorboard parameters
    parser.add_argument("-nn_tb_dir", type=str, default='SPO_TENSORBOARD')

    # Architecture parameters
    parser.add_argument("-nn_depth", type=int, default=0)
    parser.add_argument("-nn_width", type=int, default=100)

    # CPLEX parameters
    parser.add_argument("-nn_cpx_timelimit", type=float, default=60)
    parser.add_argument("-nn_cpx_threads", type=int, default=1)

    # Warmstart parameters
    parser.add_argument("-nn_warmstart_dir", type=str, default='')
    parser.add_argument("-nn_warmstart_prefix", type=str, default='')

    # Data parameters
    parser.add_argument("-nn_poly_degree", type=int, default=1)

    args = parser.parse_args(args)
    print(args)

    # output directories
    output_dir = "SPO_MODELS/" + args.output_dir
    try: 
        os.makedirs(output_dir)
    except OSError:
        if not os.path.exists(output_dir):
            raise

    time_dataread, time_train, time_validation = 0.0, 0.0, 0.0
    mse_train, r2_train = 0.0, 0.0
    mse_validation, r2_validation = 0.0, 0.0

    validation_bool = len(args.data_validation_dir) > 0

    if args.method == '2stage':
        # aggregate data in args.data_train_dir
        time_dataread = time.time()
        data_train_full, _ = combine_datasets(args.data_train_dir)
        num_features = len(data_train_full[0][0,:]) - 2

        if args.single_model:
            data_train_full[:,0] = 0

        if validation_bool:
            data_validation_full, _ = combine_datasets(args.data_validation_dir)

            if args.single_model:
                data_validation_full[:,0] = 0

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

            if validation_bool:
                rows_indicator = np.where(data_validation_full[:,0] == indicator)[0]
                X_validation, y_validation = data_validation_full[rows_indicator,1:num_features+1], data_validation_full[rows_indicator,-1]

                time_validation, mse_validation, r2_validation = test_model(regr, X_validation, y_validation)

            results_str = "%d,%s,%g,%g,%g,%g,%g,%g,%g,%s" % (
                indicator,
                args.model_type,
                time_train,
                mse_train, 
                r2_train,
                time_validation,
                mse_validation, 
                r2_validation,
                time_dataread,
                model_filename)

            print(results_str)

            results_filename = '%s/%s_%d_%d.csv' % (output_dir, args.model_type, args.poly_degree, indicator)
            with open(results_filename, "w+") as results_file:
                results_file.write(results_str)

    elif args.method == 'spo':
        np.random.seed(0)
        torch.manual_seed(0)
        warmstart_bool = int(args.nn_warmstart_dir != '' and args.nn_warmstart_prefix != '')

        stages = ['train']

        num_epochs = args.nn_epochs
        model_indicators = [0,1]

        dtype = torch.double
        device = torch.device("cpu")
        torch.set_default_dtype(dtype)

        filename_noext = 'depth_%d_width_%d_reg_%g_polydeg_%d_lrinit_%g_lrdecay_%d_warmst_%d' % (
            args.nn_depth, 
            args.nn_width,
            args.nn_reg,
            args.nn_poly_degree, 
            args.nn_lr_init,
            args.nn_lr_decay,
            warmstart_bool)
        model_filename = '%s/%s.pt' % (output_dir, filename_noext)

        # Tensorboard setup
        tb_dirname = '%s/%s/%s' % (args.nn_tb_dir, args.output_dir, filename_noext)
        try: 
            os.makedirs(tb_dirname)
        except OSError:
            if not os.path.exists(tb_dirname):
                raise
        tb_writer = SummaryWriter(tb_dirname)

        meta_dict = {}

        data_train_full, data_train_filenames = combine_datasets(args.data_train_dir, operation='list', poly_degree=args.nn_poly_degree)
        meta_dict['train'] = build_dict(data_train_filenames, data_train_full, args)
        num_features = len(data_train_full[0][0,:]) - 2

        if validation_bool:
            data_validation_full, data_validation_filenames = combine_datasets(args.data_validation_dir, operation='list', poly_degree=args.nn_poly_degree)
            meta_dict['validation'] = build_dict(data_validation_filenames, data_validation_full, args)
            stages += ['validation']

        loss_fn = SPOLoss.apply
        depth, width = args.nn_depth, args.nn_width
        models = [SPONet(num_features, depth, width, relu_sign=-1), SPONet(num_features, depth, width, relu_sign=1)]
        print(models[1].layers[0].weight.data)

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

        running_loss_best, running_loss_withreg_best, epoch_best = np.inf, np.inf, -1
        time_solve = 0.0
        for epoch in range(num_epochs):
            sys.stdout.flush()
            for stage in stages:
                if stage == 'validation':
                    for model in models:
                        model.eval()

                    batchsize = meta_dict['validation']['num_instances']

                if stage == 'train':
                    print('---------------')
                    print("Learning rate =", optimizer.param_groups[0]['lr'])
                    if args.nn_lr_decay:
                        optimizer.param_groups[0]['lr'] = lmbda(epoch)

                    batchsize = args.nn_batchsize

                perm = torch.tensor(np.random.permutation(range(meta_dict[stage]['num_instances'])))
                batches = torch.split(perm, batchsize)

                print(batches)

                running_loss, running_loss_withreg = 0.0, 0.0

                for batch_cur_ in batches:
                    optimizer.zero_grad()
                    batch_cur = batch_cur_.tolist()
                    print(batch_cur)
                    
                    coeffs = []
                    for instance_idx_ in batch_cur:
                        instance_idx = int(instance_idx_)
                        print(instance_idx)
                        indices = meta_dict[stage]['model_indices'][instance_idx]
                        coeffs_cur = [models[0](meta_dict[stage]['data'][instance_idx][indices[0], 1:num_features+1]), models[1](meta_dict[stage]['data'][instance_idx][indices[1], 1:num_features+1])]
                        coeffs += [torch.cat((coeffs_cur[0], coeffs_cur[1]), 0)]

                    def solveIP_obj(idx):
                        instance_idx = batch_cur[idx]

                        obj_subgradient = (2*coeffs[idx] - meta_dict[stage]['coeffs_true'][instance_idx]) / (torch.max(torch.abs(coeffs[idx])) + 1)
                        time_cur = time.time()
                        sol_spo_cur, _ = spo_torch.solveIP_obj(meta_dict[stage]['instance_cpx'][instance_idx], obj_subgradient)
                        time_cur = time.time() - time_cur

                        return (sol_spo_cur, time_cur)

                    ret_vals = []
                    if args.nn_poolsize > 1 and args.nn_batchsize > 1:
                        with mp.Pool(args.nn_poolsize) as p:
                            ret_vals = p.map(solveIP_obj, range(len(batch_cur)))
                    else:
                        for idx in range(len(batch_cur)):
                            sol_spo_cur, time_cur = solveIP_obj(idx)
                            ret_vals += [(sol_spo_cur, time_cur)]

                    for idx, ret in enumerate(ret_vals):
                        instance_idx = batch_cur[idx]
                        loss_val_cur = loss_fn(
                            coeffs[idx],
                            meta_dict[stage]['coeffs_true'][instance_idx],
                            ret[0], 
                            meta_dict[stage]['sol_true'][instance_idx])

                        loss_spo_cur = float(loss_val_cur - meta_dict[stage]['objval_true'][instance_idx])
                        loss_spo_cur_scaled = loss_spo_cur / ((1e-9 + np.abs(meta_dict[stage]['objval_true'][instance_idx])) * meta_dict[stage]['num_instances'])

                        print("%s -- SPO loss [%d-%d] = %g = (%g) - (%g)" % (stage, epoch, instance_idx, loss_spo_cur, loss_val_cur, meta_dict[stage]['objval_true'][instance_idx]))

                        loss_val_cur += args.nn_reg * torch.norm(coeffs[idx])

                        running_loss_withreg += loss_val_cur.data / meta_dict[stage]['num_instances']
                        running_loss += loss_spo_cur_scaled
                        time_solve += ret[1]

                        loss_val_cur /= 1.0*len(batch_cur)

                        if stage == 'train':
                            loss_val_cur.backward() 
                    if stage == 'train':
                        optimizer.step()

                tb_writer.add_scalar('%s loss' % stage, running_loss, epoch)
                if stage == 'train':
                    tb_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
                    tb_writer.add_scalar('solver time', time_solve, epoch)
                    tb_writer.add_histogram('model0 gradients', models[0].layers[0].weight.grad, epoch)
                    tb_writer.add_histogram('model1 gradients', models[1].layers[0].weight.grad, epoch)
                    tb_writer.add_histogram('model0 weights', models[0].layers[0].weight.data, epoch)
                    tb_writer.add_histogram('model1 weights', models[1].layers[0].weight.data, epoch)

                if stage == 'validation':
                    if running_loss < running_loss_best:# or running_loss_withreg < running_loss_withreg_best:
                        running_loss_best = running_loss
                        running_loss_withreg_best = running_loss_withreg
                        epoch_best = epoch

                        torch.save({
                            'epoch': epoch, 
                            'num_features': num_features,
                            'nn_poly_degree': args.nn_poly_degree,
                            'depth': depth,
                            'width': width,
                            'model0_state_dict': models[0].state_dict(),
                            'model1_state_dict': models[1].state_dict(),
                            'loss_spo': running_loss_best,
                            'loss_spo_withreg': running_loss_withreg_best
                            }, model_filename)

                    if epoch - epoch_best >= args.nn_patience or running_loss_best <= args.nn_termination:
                        print("Early termination!")
                        print("epoch - epoch_best =", epoch - epoch_best)
                        print("running_loss_best =", running_loss_best)

                        return    

if __name__ == '__main__':
    main(sys.argv[1:])

