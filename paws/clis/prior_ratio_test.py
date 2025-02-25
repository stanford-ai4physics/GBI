__all__ = ["prior_ratio_test"]

import os
import json

import numpy as np
import pandas as pd
import click

from quickstats.maths.numerics import get_nbatch, str_encode_value
from quickstats.utils.common_utils import NpEncoder

from .main import cli

def get_components(m1: float, m2: float, mu: float, alpha: float, **kwargs):
    from paws.components import ModelTrainer, ModelLoader
    model_options = {
        "mass_point": [m1, m2],
        "mu": mu,
        "alpha": alpha
    }
    model_trainer = ModelTrainer("semi_weakly", model_options=model_options, decay_modes="qq,qqq",
                                 cache_test=True, variables="3,5,6", **kwargs)
    datasets = model_trainer.get_datasets()
    ws_model = model_trainer.get_model()
    fs_2_model = model_trainer.model_loader.fs_2_model
    fs_3_model = model_trainer.model_loader.fs_3_model

    # set to correct weight first
    weights = {
        'm1': m1 / 100,
        'm2': m2 / 100,
        'mu': -1,
        'alpha': alpha
    }
    
    ModelLoader.set_model_weights(ws_model, weights)
    
    x = np.concatenate([d[0][0] for d in datasets['train']])
    y_true = np.concatenate([d[1] for d in datasets['train']]).flatten()
    y_pred_2 = fs_2_model.predict(x, batch_size=1024).flatten()
    y_pred_3 = fs_3_model.predict(x, batch_size=1024).flatten()


    models = {
        'ws': ws_model,
        'fs2': fs_2_model,
        'fs3': fs_3_model,
        'prior_ratio': {
            'qq' : {},
            'qqq': {}
        }
    }
    
    outputs = {
        'x': x,
        'y_true': y_true,
        'y_pred_2': y_pred_2,
        'y_pred_3': y_pred_3
    }

    model_options = {
        'sampling_method': 'sampled'
    }
    
    qq_prior_trainer = ModelTrainer("prior_ratio", model_options=model_options, decay_modes="qq",
                                    cache_test=True, variables="3,5,6", **kwargs)
    
    qqq_prior_trainer = ModelTrainer("prior_ratio", model_options=model_options, decay_modes="qqq",
                                     cache_test=True, variables="3,5,6", **kwargs)
    models['prior_ratio']['qq']['sampled'] = qq_prior_trainer.load_trained_model()
    models['prior_ratio']['qqq']['sampled'] = qqq_prior_trainer.load_trained_model()
    
    model_options = {
        'sampling_method': 'inferred'
    }
    qq_prior_trainer = ModelTrainer("prior_ratio", model_options=model_options, decay_modes="qq",
                                    cache_test=True, variables="3,5,6", **kwargs)
    qqq_prior_trainer = ModelTrainer("prior_ratio", model_options=model_options, decay_modes="qqq",
                                     cache_test=True, variables="3,5,6", **kwargs)
    models['prior_ratio']['qq']['inferred'] = qq_prior_trainer.load_trained_model()
    models['prior_ratio']['qqq']['inferred'] = qqq_prior_trainer.load_trained_model()

    return models, outputs

def get_bce(y_true, y_pred_2, y_pred_3, mu, alpha, kappa_2=1.0, kappa_3=1.0, vectorize: bool = False):
    L2 = kappa_2 * y_pred_2 / (1 - y_pred_2)
    L3 = kappa_3 * y_pred_3 / (1 - y_pred_3)
    ws = (mu * (alpha * L3  + (1 - alpha) * L2 ) + 1 - mu) / (mu * (alpha * L3 + (1 - alpha) * L2) +2 - 2 *mu)
    axis = 1 if vectorize else None
    return (np.mean(-(y_true * np.log(ws) + (1 - y_true) * np.log(1 - ws)), axis=axis) - np.log(2)) * 1e5

def run_loss_scan(mu_arr, alpha_arr,
                  y_true, y_pred_2, y_pred_3,
                  kappa_2, kappa_3,
                  batchsize: int = 500):
    alpha_grid, mu_grid = np.meshgrid(alpha_arr, mu_arr)
    mu_grid = mu_grid.reshape(-1, 1)
    alpha_grid = alpha_grid.reshape(-1, 1)

    results = {
        'mu': np.array([]),
        'alpha': np.array([]),
        'loss': np.array([])
    }
    
    y_pred_2_all = None
    y_pred_3_all = None
    y_true_all = None
    
    nbatch = get_nbatch(mu_grid.shape[0], batchsize)
    for i in range(nbatch):
        print(f'Batch {i+1} / {nbatch}')
        mu_sub = mu_grid[i * batchsize : (i + 1) * batchsize]
        alpha_sub = alpha_grid[i * batchsize : (i + 1) * batchsize]
        if (y_pred_2_all is None) or (y_pred_2_all.shape[0] != mu_sub.shape[0]):
            y_pred_2_all = np.tile(y_pred_2, (mu_sub.shape[0], 1))
            y_pred_3_all = np.tile(y_pred_3, (mu_sub.shape[0], 1))
            y_true_all = np.tile(y_true, (mu_sub.shape[0], 1))
        losses = get_bce(y_true_all, y_pred_2_all, y_pred_3_all,
                         mu_sub, alpha_sub, kappa_2=kappa_2, kappa_3=kappa_3,
                         vectorize=True)
        results['mu'] = np.concatenate([results['mu'], mu_sub.flatten()])
        results['alpha'] = np.concatenate([results['alpha'], alpha_sub.flatten()])
        results['loss'] = np.concatenate([results['loss'], losses.flatten()])
    return results

def run_scenario(m1: float, m2: float, mu: float, alpha: float,
                 batchsize:int = 500, **kwargs):
    models, outputs = get_components(m1=m1, m2=m2, mu=mu, alpha=alpha,
                                     **kwargs)
    y_true = outputs['y_true']
    y_pred_2 = outputs['y_pred_2']
    y_pred_3 = outputs['y_pred_3']
    all_results = {
    }
    alpha_arr = np.arange(0, 1, 0.01)

    if mu > 0.4:
        mu_arr = np.arange(0.1, 0.8 + 0.1, 0.01)
    elif mu > 0.1:
        mu_arr = np.logspace(-1, -0.2, 60)
    else:
        mu_arr = np.exp(np.arange(-8, -2, 0.1))

    #kappa_2, kappa_3 = 1.0, 1.0
    #print(f'Running with kappa_2 = {kappa_2}, kappa_3 = {kappa_3}')
    #results = run_loss_scan(mu_arr=mu_arr, alpha_arr=alpha_arr,
    #                        y_true=y_true, y_pred_2=y_pred_2,
    #                        y_pred_3=y_pred_3, kappa_2=kappa_2,
    #                        kappa_3=kappa_3, batchsize=batchsize)
    #all_results['no_kappa'] = results
    #
    #kappa_2, kappa_3 = 10., 10.
    #print(f'Running with kappa_2 = {kappa_2}, kappa_3 = {kappa_3}')
    #results = run_loss_scan(mu_arr=mu_arr, alpha_arr=alpha_arr,
    #                        y_true=y_true, y_pred_2=y_pred_2,
    #                        y_pred_3=y_pred_3, kappa_2=kappa_2,
    #                        kappa_3=kappa_3, batchsize=batchsize)
    #all_results['uniform_kappa'] = results
    #model_2, model_3 = models['prior_ratio']['qq']['inferred'], models['prior_ratio']['qqq']['inferred']
    #kappa_2, kappa_3 = model_2.predict([[m1, m2]])[0][0], model_3.predict([[m1, m2]])[0][0]
    #print(f'Running with kappa_2 = {kappa_2}, kappa_3 = {kappa_3}')
    #results = run_loss_scan(mu_arr=mu_arr, alpha_arr=alpha_arr,
    #                        y_true=y_true, y_pred_2=y_pred_2,
    #                        y_pred_3=y_pred_3, kappa_2=kappa_2,
    #                        kappa_3=kappa_3, batchsize=batchsize)
    #all_results['inferred_kappa'] = results

    model_2, model_3 = models['prior_ratio']['qq']['sampled'], models['prior_ratio']['qqq']['sampled']
    kappa_2, kappa_3 = model_2.predict([[m1, m2]])[0][0], model_3.predict([[m1, m2]])[0][0]
    print(f'Running with kappa_2 = {kappa_2}, kappa_3 = {kappa_3}')
    results = run_loss_scan(mu_arr=mu_arr, alpha_arr=alpha_arr,
                            y_true=y_true, y_pred_2=y_pred_2,
                            y_pred_3=y_pred_3, kappa_2=kappa_2,
                            kappa_3=kappa_3, batchsize=batchsize)
    all_results['sampled_kappa'] = results
    return all_results

@cli.command(name='prior_ratio_test')
@click.option('--m1', required=True, type=float,
              help='Value of m1 for the signal sample.')
@click.option('--m2', required=True, type=float,
              help='Value of m2 for the signal sample.')
@click.option('--mu', required=True, type=float,
              help='Signal fraction used in the dataset.')
@click.option('--alpha', default=0.5, type=float, show_default=True,
              help='Signal branching fraction in the dataset.')
@click.option('--batchsize', default=500, type=int, show_default=True,
              help='Batch size used for batched loss computation.')
@click.option('-d', '--datadir', default="/pscratch/sd/c/chlcheng/projects/paws/datasets", show_default=True,
              help='Input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default="/pscratch/sd/c/chlcheng/projects/paws/outputs", show_default=True,
              help='Base output directory')
@click.option('--dataset-index-path', 'index_path', default="/pscratch/sd/c/chlcheng/projects/paws/semiweakly_dataset_indices.json", show_default=True,
              help='\b\n Path to the dataset split configuration file. It determines the'
              '\b\n shard indices for the train, validation, and test datasets in each'
              '\b\n random realization of data. If None, a default configuration '
              '\b\n will be created.')
@click.option('-i', '--split-index', default=0, type=int, show_default=True,
              help='Index for dataset split.')
def prior_ratio_test(**kwargs):
    result = run_scenario(**kwargs)
    result['truth'] = {
        'm1': kwargs["m1"],
        'm2': kwargs["m2"],
        'mu': kwargs["mu"],
        'alpha': kwargs["alpha"]
    }
    outname = os.path.join(kwargs['outdir'], 'prior_ratio_test',
                           f'{str_encode_value(kwargs["m1"])}_'
                           f'{str_encode_value(kwargs["m2"])}_'
                           f'{str_encode_value(kwargs["mu"])}_'
                           f'{str_encode_value(kwargs["alpha"])}.json')
    with open(outname, 'w') as outfile:
        json.dump(result, outfile, cls=NpEncoder)