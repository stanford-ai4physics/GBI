from typing import Tuple, Optional, List, Dict, Any
import os
import json

import click
import numpy as np

from quickstats.utils.string_utils import split_str

from paws.settings import DEFAULT_DATADIR, DEFAULT_OUTDIR, BASE_SEED, MASS_RANGE, MASS_INTERVAL, ModelType
from .main import cli, DelimitedStr

__all__ = ["compute_semi_weakly_landscape", "gather_model_results"]
    
kCommonKeys = ["high_level", "decay_modes", "variables", "noise_dimension", "version",
               "index_path", "split_index", "seed", "batchsize", "cache_dataset",
               "datadir", "outdir", "cache", "multi_gpu", "verbosity", "loss",
               "initial_check"]

def get_model_trainer(model_type:str, **kwargs):
    from paws.components import ModelTrainer
    init_kwargs = {key: kwargs.pop(key) for key in kCommonKeys if key in kwargs}
    # dedicated supervised
    if ('signal' in kwargs) and ('background' in kwargs):
        sig_samples = list(set(split_str(kwargs.pop('signal'), sep=',', remove_empty=True)))
        bkg_samples = list(set(split_str(kwargs.pop('background'), sep=',', remove_empty=True)))
        if not set(sig_samples).isdisjoint(set(bkg_samples)):
            raise ValueError('Signal and background samples must be disjoint')
        kwargs['samples'] = sig_samples + bkg_samples
        kwargs['supervised_label_map'] = {
            **{sample: 0 for sample in bkg_samples},
            **{sample: 1 for sample in sig_samples},
        }
    # weakly
    elif ('signal' in kwargs) and ('data_background' in kwargs):
        sig_samples = list(set(split_str(kwargs.pop('signal'), sep=',', remove_empty=True)))
        dat_bkg_samples = list(set(split_str(kwargs.pop('data_background'), sep=',', remove_empty=True)))
        ref_bkg_samples = list(set(split_str(kwargs.pop('reference_background'), sep=',', remove_empty=True)))
        bkg_samples = list(set(dat_bkg_samples + ref_bkg_samples))
        if not set(sig_samples).isdisjoint(set(bkg_samples)):
            raise ValueError('Signal and background samples must be disjoint')
        kwargs['samples'] = sig_samples + bkg_samples
        kwargs['supervised_label_map'] = {
            **{sample: 0 for sample in bkg_samples},
            **{sample: 1 for sample in sig_samples},
        }
        kwargs['weakly_label_map'] = {}
        for sample in bkg_samples:
            if (sample in dat_bkg_samples) and (sample in ref_bkg_samples):
                kwargs['weakly_label_map'][sample] = -1
            elif sample in ref_bkg_samples:
                kwargs['weakly_label_map'][sample] = 0
            elif sample in dat_bkg_samples:
                kwargs['weakly_label_map'][sample] = 1
        weakly_labels = list(kwargs['weakly_label_map'].values())
        if (-1 in weakly_labels) and ((0 in weakly_labels) or (1 in weakly_labels)):
            raise ValueError(
                'Same sample cannot enter both data background and reference background unless '
                'every sample does so'
            )
    # prior ratio
    elif ('background' in kwargs):
        kwargs['samples'] = list(set(split_str(kwargs.pop('background'), sep=',', remove_empty=True)))    
    init_kwargs['model_options'] = kwargs
    feature_level = "high_level" if init_kwargs.pop("high_level") else "low_level"
    model_trainer = ModelTrainer(model_type, **init_kwargs)
    return model_trainer

DEFAULT_PARAM_EXPR = (f'm1=0_{MASS_RANGE[1]}_{MASS_INTERVAL},'
                      f'm2=0_{MASS_RANGE[1]}_{MASS_INTERVAL}')

@cli.command(name='compute_supervised_landscape')
@click.option('-m', '--mass-point', required=True,
              help='Signal mass point (in the form m1:m2) to use for creating the dataset.')
@click.option('--version', 'version', required=True,
              help='Version of the supervised model to use.')
@click.option('--param-expr', default=DEFAULT_PARAM_EXPR, show_default=True,
              help='\b\n An expression specifying the parameter space to scan over.'
              '\b\n The format is "<param_name>=<min_val>_<max_val>_<step>".'
              '\b\n Multi-dimensional space can be specified by joining two'
              '\b\n expressions with a comma. To fix the value of a parameter,'
              '\b\n use the format "<param_name>=<value>". To includ a finite'
              '\b\n set of values, use "<param_name>=(<value_1>,<value_2>,...)".')
@click.option('--samples', default=None,
              show_default=True,
              help='Samples to be used for evaluation')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq', type=click.Choice(['qq', 'qqq'], case_sensitive=False), show_default=True,
              help='Which decay mode should the signal undergo (qq or qqq).')
@click.option('--variables', default=None, show_default=True,
              help='Select certain high-level jet features to include in the training'
              'by the indices they appear in the feature vector. For example,'
              '"3,5,6" means select the 4th, 6th and 7th feature from the jet'
              'feature vector to be used in the training.')
@click.option('--noise', 'noise_dimension', default=0, type=int, show_default=True,
              help='Number of noise dimension to add to the train features.')
@click.option('--metrics', default=None, show_default=True,
              help='\b\n List of metrics to evaluate. If None, the model output as'
              '\b\n well as the truth labels will be saved instead.')
@click.option('--dataset-index-path', 'index_path', default=None, show_default=True,
              help='\b\n Path to the dataset split configuration file. It determines the'
              '\b\n shard indices for the train, validation, and test datasets in each'
              '\b\n random realization of data. If None, a default configuration '
              '\b\n will be created.')
@click.option('-i', '--split-index', default=0, type=int, show_default=True,
              help='Index for dataset split.')
@click.option('--ds-type', default='test', type=click.Choice(['train', 'val', 'test'], case_sensitive=False), show_default=True,
              help='Type of dataset to use for evaluation (choose between train, val and test).')
@click.option('--tag', default='default', show_default=True,
              help='Extra tag added to the output directory tree.')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for the dataset.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset during training.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='base output directory')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the results.')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU computation.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def compute_supervised_landscape(**kwargs):
    """
    Compute metric landscapes for a supervised model
    """
    import os
    import json
    import numpy as np
    from quickstats.utils.common_utils import NpEncoder
    from quickstats.utils.string_utils import split_str
    from paws.components import MetricLandscape

    ds_type = kwargs.pop('ds_type')
    param_expr = kwargs.pop('param_expr')
    metrics = kwargs.pop('metrics')
    mass_point = kwargs.pop('mass_point')
    samples = kwargs.pop('samples')
    tag = kwargs.pop('tag')
    if metrics:
        metrics = split_str(metrics, sep=',', remove_empty=True)
    if samples:
        samples = split_str(samples, sep=',', remove_empty=True)

    model_trainer = get_model_trainer("param_supervised", **kwargs)
    mass_point = model_trainer.parse_mass_point(mass_point)
    parameters = model_trainer.model_loader._get_param_repr()
    model_options = model_trainer.model_options
    parameters['mass_point'] = mass_point
    parameters['split_index'] = kwargs['split_index']
    parameters = model_trainer.path_manager.process_parameters(**parameters)
    parameters['tag'] = tag
    outname = model_trainer.path_manager.get_file("supervised_landscape",
                                                   **parameters,
                                                   ds_type=ds_type)

    if kwargs['cache'] and os.path.exists(outname):
        model_trainer.stdout.info(f"Cached semi-weakly model landscape output from {outname}")
        return
    model = model_trainer.load_trained_model()

    datasets = model_trainer.get_datasets(
        mass_point=mass_point,
        samples=samples
    )
    
    landscape = MetricLandscape()
    result = landscape.eval_supervised(
        model,
        datasets[ds_type],
        param_expr=param_expr,
        metrics=metrics
    )
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    with open(outname, 'w') as file:
        json.dump(result, file, cls=NpEncoder)
    model_trainer.stdout.info(f"Saved supervised model landscape output to {outname}")

@cli.command(name='compute_semi_weakly_landscape')
@click.option('-m', '--mass-point', required=True,
              help='Signal mass point (in the form m1:m2) to use for creating the dataset.')
@click.option('--param-expr', default=DEFAULT_PARAM_EXPR, show_default=True,
              help='\b\n An expression specifying the parameter space to scan over.'
              '\b\n The format is "<param_name>=<min_val>_<max_val>_<step>".'
              '\b\n Multi-dimensional space can be specified by joining two'
              '\b\n expressions with a comma. To fix the value of a parameter,'
              '\b\n use the format "<param_name>=<value>". To includ a finite'
              '\b\n set of values, use "<param_name>=(<value_1>,<value_2>,...)".')
@click.option('--metrics', default=None, show_default=True,
              help='\b\n List of metrics to evaluate. If None, the model output as'
              '\b\n well as the truth labels will be saved instead.')
@click.option('--mu', required=True, type=float,
              help='Signal fraction used in the dataset.')
@click.option('--alpha', default=0.5, type=float, show_default=True,
              help='Signal branching fraction in the dataset. Ignored '
             'when only one signal decay mode is considered.')
@click.option('--kappa', default='1.0', type=str, show_default=True,
              help='Prior normalization factor. It can be a number (fixing kappa value), or a string '
              '. If string, it should be either "sampled" (kappa learned from sampling) or '
              '"inferred" (kappa learned from event number).')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False), show_default=True,
              help='Which decay mode should the signal undergo (qq or qqq).'
              'Use "qq,qqq" to include both decay modes.')
@click.option('--use-trained-weight/--use-true-weight', default=False, show_default=True,
              help='Whether to use the trained model to initialize the weight.')
@click.option('--train-data-size', default=None, type=int, show_default=True,
          help='Restrict the number of background events as data (label = 1) used in training.')
@click.option('--val-data-size', default=None, type=int, show_default=True,
          help='Restrict the number of background events as data (label = 1) used in validation.')
@click.option('--test-data-size', default=None, type=int, show_default=True,
          help='Restrict the number of background events as data (label = 1) used in testing.')
@click.option('--variables', default=None, show_default=True,
              help='Select certain high-level jet features to include in the training'
              'by the indices they appear in the feature vector. For example,'
              '"3,5,6" means select the 4th, 6th and 7th feature from the jet'
              'feature vector to be used in the training.')
@click.option('--signal', default='W_qq,W_qqq',
              show_default=True,
              help='Signal samples to be used in weakly dataset mixing.')
@click.option('--data-background', default='QCD,extra_QCD',
              show_default=True,
              help='Data background samples to be used in weakly dataset mixing.')
@click.option('--reference-background', default='QCD,extra_QCD',
              show_default=True,
              help='Reference background samples to be used in weakly dataset mixing.')
@click.option('--noise', 'noise_dimension', default=0, type=int, show_default=True,
              help='Number of noise dimension to add to the train features.')
@click.option('--loss', default='bce', type=click.Choice(['bce', 'nll'], case_sensitive=False),
              show_default=True,
              help='\b\n Name of the loss function. Choose between "bce" (binary '
              '\b\n cross entropy) and "nll" (negative log-likelihood).')
@click.option('--dataset-index-path', 'index_path', default=None, show_default=True,
              help='\b\n Path to the dataset split configuration file. It determines the'
              '\b\n shard indices for the train, validation, and test datasets in each'
              '\b\n random realization of data. If None, a default configuration '
              '\b\n will be created.')
@click.option('-i', '--split-index', default=0, type=int, show_default=True,
              help='Index for dataset split.')
@click.option('--fs-split-index', default=None, type=int, show_default=True,
              help='Index for prior model dataset split. Use -1 to include prior '
              'models from all dataset splits. If None, the same dataset split index '
              'as the semi-weakly model will be used.')
@click.option('--ds-type', default='test', type=click.Choice(['train', 'val', 'test'], case_sensitive=False), show_default=True,
              help='Type of dataset to use for evaluation (choose between train, val and test).')
@click.option('--version', default="v1", show_default=True,
              help='Version of the model.')
@click.option('--fs-version', 'fs_version', default="v1", show_default=True,
              help='Version of the supervised model to use.')
@click.option('--fs-version-2', 'fs_version_2', default=None, show_default=True,
              help='\b\n When signals of mixed decay modes are considered, it corresponds to '
             '\b\n the version of the three-prone supervised model. If None, the '
             '\b\n same version as `fs_version` will be used.')
@click.option('--tag', default='default', show_default=True,
              help='Extra tag added to the output directory tree.')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='The default seed used for all random processes.')
@click.option('--nbootstrap', default=None, type=int, show_default=True,
              help='Number of bootstrap samples.')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for the dataset.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset during training.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='base output directory')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the results.')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU computation.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def compute_semi_weakly_landscape(**kwargs):
    """
    Compute metric landscapes for a semi-weakly model
    """
    import os
    import json
    import numpy as np
    from quickstats.utils.common_utils import NpEncoder
    from quickstats.utils.string_utils import split_str
    from paws.components import MetricLandscape

    ds_type = kwargs.pop('ds_type')
    use_trained_weight = kwargs.pop('use_trained_weight')
    param_expr = kwargs.pop('param_expr')
    metrics = kwargs.pop('metrics')
    nbootstrap = kwargs.pop('nbootstrap')
    seed = kwargs['seed']
    tag = kwargs.pop('tag')
    if metrics:
        metrics = split_str(metrics, sep=',', remove_empty=True)

    kwargs['weakly_test'] = True
    model_trainer = get_model_trainer("semi_weakly", **kwargs)
    parameters = model_trainer.model_loader._get_param_repr()
    model_options = model_trainer.model_options
    parameters['mass_point'] = model_options["mass_point"]
    parameters['mu'] = model_options["mu"]
    parameters['alpha'] = model_options["alpha"]
    parameters['split_index'] = kwargs['split_index']
    parameters = model_trainer.path_manager.process_parameters(**parameters)
    parameters['tag'] = tag
    outname = model_trainer.path_manager.get_file("semi_weakly_landscape",
                                                   **parameters,
                                                   ds_type=ds_type)
    if kwargs['cache'] and os.path.exists(outname):
        model_trainer.stdout.info(f"Cached semi-weakly model landscape output from {outname}")
        return
    datasets = model_trainer.get_datasets()
    if use_trained_weight:
        model = model_trainer.load_trained_model()
    else:
        model = model_trainer.get_model()
    landscape = MetricLandscape()

    result = landscape.eval_semiweakly(
        model, datasets[ds_type],
        param_expr=param_expr,
        metrics=metrics,
        nbootstrap=nbootstrap,
        seed=seed
    )
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    with open(outname, 'w') as file:
        json.dump(result, file, cls=NpEncoder)
    model_trainer.stdout.info(f"Saved semi-weakly model landscape output to {outname}")

def get_nll(
    y_true,
    llr_2,
    mu,
    alpha = None,
    llr_3 = None,
    vectorize: bool = False,
    poisson_seed = None,
    epsilon: float = 1e-10
):
    if llr_3 is None:
        llr_xs = 1. + mu * (llr_2 - 1.)
    else:
        llr_xs = 1. + mu * (alpha * llr_3 + (1 - alpha) * llr_2 - 1.)
    llr_xs = np.clip(llr_xs, epsilon, np.inf)
    axis = 1 if vectorize else None
    if poisson_seed is not None:
        np.random.seed(poisson_seed)
        poisson_weight = np.random.poisson(size=y_true.shape[1])
        return - np.sum(y_true * poisson_weight * np.log(llr_xs), axis=axis)
    return - np.sum(y_true * np.log(llr_xs), axis=axis)

def run_likelihood_scan(
    mu_arr,
    y_true,
    llr_2,
    llr_3 = None,
    alpha_arr = None,
    batchsize: int = 500,
    poisson_seeds: Optional[List[int]] = None
):
    from quickstats.maths.numerics import get_nbatch

    results = {
        'mu': [],
        'loss': [],
        'seed': []
    }

    if (llr_3 is not None) and (alpha_arr is None):
        raise ValueError('`alpha_arr` must be specified if `llr_3` is given')
    
    if llr_3 is None:
        mu_grid = mu_arr
        mu_grid = mu_grid.reshape(-1, 1)
        alpha_grid = None
    else:
        alpha_grid, mu_grid = np.meshgrid(alpha_arr, mu_arr)
        mu_grid = mu_grid.reshape(-1, 1)
        alpha_grid = alpha_grid.reshape(-1, 1)
        results['alpha'] = []

    llr_2_all = None
    llr_3_all = None
    y_true_all = None
    
    nbatch = get_nbatch(mu_grid.shape[0], batchsize)
    poisson_seeds = poisson_seeds or [None]
    for i in range(nbatch):
        print(f'Batch {i+1} / {nbatch}')
        mu_sub = mu_grid[i * batchsize : (i + 1) * batchsize]
        alpha_sub = None if llr_3 is None else alpha_grid[i * batchsize : (i + 1) * batchsize]
        if (llr_2_all is None) or (llr_2_all.shape[0] != mu_sub.shape[0]):
            llr_2_all = np.tile(llr_2, (mu_sub.shape[0], 1))
            llr_3_all = None if llr_3 is None else np.tile(llr_3, (mu_sub.shape[0], 1))
            y_true_all = np.tile(y_true, (mu_sub.shape[0], 1))
        for poisson_seed in poisson_seeds:
            print(f'Running on seed: {poisson_seed}')
            losses = get_nll(
                mu=mu_sub,
                alpha=alpha_sub,
                y_true=y_true_all,
                llr_2=llr_2_all,
                llr_3=llr_3_all,
                vectorize=True,
                poisson_seed=poisson_seed
            )
            results['mu'].append(mu_sub.flatten())
            if llr_3 is not None:
                results['alpha'].append(alpha_sub.flatten())
            results['loss'].append(losses.flatten())
            seed_value = -1 if poisson_seed is None else poisson_seed
            results['seed'].append(np.full(mu_sub.shape[0], seed_value))
    for key, values in results.items():
        results[key] = np.concatenate(values)
    return results

@cli.command(name='compute_semi_weakly_mu_alpha_likelihood_landscape')
@click.option('-m', '--mass-point', required=True,
              help='Signal mass point (in the form m1:m2) to use for creating the dataset.')
@click.option('--mu-expr', required=True,
              help='\b\n An expression specifying the mu parameter space to scan over.'
              '\b\n The format is "<min_val>_<max_val>_<step>". To fix the value'
              '\b\n of the parameter, use the format "<value>". To includ a finite'
              '\b\n set of values, use "(<value_1>,<value_2>,...).')
@click.option('--alpha-expr', default=None, show_default=True,
              help='\b\n An expression specifying the alpha parameter space to scan over.'
              '\b\n The format is "<min_val>_<max_val>_<step>". To fix the value'
              '\b\n of the parameter, use the format "<value>". To includ a finite'
              '\b\n set of values, use "(<value_1>,<value_2>,...).')
@click.option('--mu', required=True, type=float,
              help='Signal fraction used in the dataset.')
@click.option('--alpha', default=0.5, type=float, show_default=True,
              help='Signal branching fraction in the dataset. Ignored '
             'when only one signal decay mode is considered.')
@click.option('--kappa', default='1.0', type=str, show_default=True,
              help='Prior normalization factor. It can be a number (fixing kappa value), or a string '
              '. If string, it should be either "sampled" (kappa learned from sampling) or '
              '"inferred" (kappa learned from event number).')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False), show_default=True,
              help='Which decay mode should the signal undergo (qq or qqq).'
              'Use "qq,qqq" to include both decay modes.')
@click.option('--use-trained-weight/--use-true-weight', default=False, show_default=True,
              help='Whether to use the trained model to initialize the weight.')
@click.option('--variables', default=None, show_default=True,
              help='Select certain high-level jet features to include in the training'
              'by the indices they appear in the feature vector. For example,'
              '"3,5,6" means select the 4th, 6th and 7th feature from the jet'
              'feature vector to be used in the training.')
@click.option('--signal', default='W_qq,W_qqq',
              show_default=True,
              help='Signal samples to be used in weakly dataset mixing.')
@click.option('--data-background', default='QCD,extra_QCD',
              show_default=True,
              help='Data background samples to be used in weakly dataset mixing.')
@click.option('--reference-background', default='QCD,extra_QCD',
              show_default=True,
              help='Reference background samples to be used in weakly dataset mixing.')
@click.option('--noise', 'noise_dimension', default=0, type=int, show_default=True,
              help='Number of noise dimension to add to the train features.')
@click.option('--train-data-size', default=None, type=int, show_default=True,
          help='Restrict the number of background events as data (label = 1) used in training.')
@click.option('--val-data-size', default=None, type=int, show_default=True,
          help='Restrict the number of background events as data (label = 1) used in validation.')
@click.option('--test-data-size', default=None, type=int, show_default=True,
          help='Restrict the number of background events as data (label = 1) used in testing.')
@click.option('--dataset-index-path', 'index_path', default=None, show_default=True,
              help='\b\n Path to the dataset split configuration file. It determines the'
              '\b\n shard indices for the train, validation, and test datasets in each'
              '\b\n random realization of data. If None, a default configuration '
              '\b\n will be created.')
@click.option('-i', '--split-index', default=None, show_default=True,
              help='Index or list of indices for dataset split (separated by commas).')
@click.option('--fs-split-index', default=None, type=int, show_default=True,
              help='Index for prior model dataset split. Use -1 to include prior '
              'models from all dataset splits. If None, the same dataset split index '
              'as the semi-weakly model will be used.')
@click.option('--ds-type', default='test', type=click.Choice(['train', 'val', 'test'], case_sensitive=False), show_default=True,
              help='Type of dataset to use for evaluation (choose between train, val and test).')
@click.option('--version', default="v1", show_default=True,
              help='Version of the model.')
@click.option('--fs-version', 'fs_version', default="v1", show_default=True,
              help='Version of the supervised model to use.')
@click.option('--fs-version-2', 'fs_version_2', default=None, show_default=True,
              help='\b\n When signals of mixed decay modes are considered, it corresponds to '
             '\b\n the version of the three-prone supervised model. If None, the '
             '\b\n same version as `fs_version` will be used.')
@click.option('--tag', default='default', show_default=True,
              help='Extra tag added to the output directory tree.')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='The default seed used for all random processes.')
@click.option('--nbootstrap', default=None, type=int, show_default=True,
              help='Number of bootstrap samples.')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for the dataset.')
@click.option('--scan-batchsize', default=500, type=int, show_default=True,
              help='Batch size for the likelihood scans.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset during training.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='base output directory')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the results.')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU computation.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def compute_semi_weakly_mu_alpha_likelihood_landscape(**kwargs):
    """
    """
    from quickstats.maths.statistics import custom_median_with_tie
    from quickstats.parsers import ParamParser
    from quickstats.utils.common_utils import NpEncoder, list_of_dict_to_dict_of_list
    from quickstats.utils.string_utils import split_str
    from paws.components import MetricLandscape, ModelLoader
    from paws.utils import get_parameter_transforms
    from collections import defaultdict

    ds_type = kwargs.pop('ds_type')
    use_trained_weight = kwargs.pop('use_trained_weight')
    mu_expr = kwargs.pop('mu_expr')
    alpha_expr = kwargs.pop('alpha_expr')
    nbootstrap = kwargs.pop('nbootstrap')
    seed = kwargs['seed']
    scan_batchsize = kwargs.pop('scan_batchsize')
    tag = kwargs.pop('tag')

    def get_arr(name:str, expr:str):
        tokens = split_str(expr, ';', remove_empty=True)
        named_exp = ';'.join([f'{name}={token }'for token in tokens])
        return np.array(list_of_dict_to_dict_of_list(list(ParamParser.parse_param_str(named_exp)))[name])
        
    mu_arr = get_arr('mu', mu_expr)
    alpha_arr = None if alpha_expr is None else get_arr('alpha', alpha_expr)

    kwargs['loss'] = 'nll'

    
    split_index = kwargs.pop('split_index')
    if split_index is None:
        from paws.components import ResultLoader
        init_kwargs = {}
        for key in ["decay_modes", "variables", "outdir"]:
            init_kwargs[key] = kwargs[key]     
        feature_level = "high_level" if kwargs["high_level"] else "low_level"
        result_loader = ResultLoader(feature_level=feature_level, **init_kwargs)
        mass_points = [[int(m) for m in split_str(kwargs["mass_point"], sep=":")]]
        df = result_loader.get_output_status(
            model_type='semi_weakly',
            mass_points=mass_points,
            version=kwargs['version']
        )
        split_indices = sorted(df[df['done']]['split_index'].unique().astype(int).tolist())
    else:
        split_indices = [int(split_index)]

    kwargs['initial_check'] = False
    kwargs['weakly_test'] = True
    
    model_trainer = get_model_trainer(
        "semi_weakly", split_index=split_indices[0], **kwargs
    )

    mixed_signal = len(model_trainer.model_loader.decay_modes) > 1
    parameters = model_trainer.model_loader._get_param_repr()
    model_options = model_trainer.model_options
    parameters['mass_point'] = model_options["mass_point"]
    parameters['mu'] = model_options["mu"]
    parameters['alpha'] = model_options["alpha"]
    if len(split_indices) == 1:
        parameters['split_index'] = str(split_indices[0])
    else:
        parameters['split_index'] = 'all'
    parameters = model_trainer.path_manager.process_parameters(**parameters)
    parameters['tag'] = tag
    outname = model_trainer.path_manager.get_file(
        "semi_weakly_landscape",
        **parameters,
        ds_type=ds_type
    )

    if kwargs['cache'] and os.path.exists(outname):
        model_trainer.stdout.info(f"Cached semi-weakly model landscape output from {outname}")
        return

    datasets = model_trainer.get_datasets()
    batchsize = model_trainer.data_loader._suggest_batchsize(kwargs['batchsize'])

    x = np.concatenate([d[0][0] for d in datasets[ds_type]])
    y_true = np.concatenate([d[1] for d in datasets[ds_type]]).flatten()
    param_transforms = get_parameter_transforms(backend='python')

    def get_ensemble_weights_and_predictions(indices: List[int]):
        ensemble_weights = defaultdict(list)
        ensemble_predictions = defaultdict(list)
        for index in indices:
            if use_trained_weight:
                model_trainer = get_model_trainer(
                    "semi_weakly", split_index=index, **kwargs
                )
                ws_model = model_trainer.load_trained_model()
                weights = ModelLoader.get_semi_weakly_model_weights(ws_model)
                m1_true, m2_true = model_trainer.model_options['mass_point']
                if m1_true != m2_true:
                    mass_weights_sorted = np.sort([weights['m1'], weights['m2']])
                    if m1_true > m2_true:
                        weights['m1'] = mass_weights_sorted[-1]
                        weights['m2'] = mass_weights_sorted[0]
                    else:
                        weights['m1'] = mass_weights_sorted[0]
                        weights['m2'] = mass_weights_sorted[-1]
            else:
                mass_point = model_options["mass_point"]
                weights = {
                    'm1': param_transforms['m1'].inverse(mass_point[0]),
                    'm2': param_transforms['m2'].inverse(mass_point[1]),
                    'mu': param_transforms['mu'].inverse(kwargs['mu'])
                }
                if mixed_signal:
                    weights['alpha'] = param_transforms['alpha'].inverse(kwargs['alpha'])
            for key, value in weights.items():
                ensemble_weights[key].append(value)
                prediction = param_transforms[key](value)
                ensemble_predictions[key].append(prediction)
        return ensemble_weights, ensemble_predictions

    def get_ensemble_likelihoods(indices: List[int], weights: Dict[str, Any]):
        ensemble_likelihoods = defaultdict(list)
        for index in indices:
            model_trainer = get_model_trainer(
                "semi_weakly", split_index=index, **kwargs
            )
            ws_model = model_trainer.get_model()
            ModelLoader.set_model_weights(ws_model, weights)
            if mixed_signal:
                llr_2_model = model_trainer.model_loader.llr_2_model
                llr_3_model = model_trainer.model_loader.llr_3_model
                llr_2 = llr_2_model.predict(x, batch_size=batchsize).flatten()
                llr_3 = llr_3_model.predict(x, batch_size=batchsize).flatten()
                ensemble_likelihoods['llr_2'].append(llr_2)
                ensemble_likelihoods['llr_3'].append(llr_3)
            else:
                llr_model = model_trainer.model_loader.llr_model
                llr = llr_model.predict(x, batch_size=batchsize).flatten()
                ensemble_likelihoods['llr_2'].append(llr)
        for key, likelihoods in ensemble_likelihoods.items():
            ensemble_likelihoods[key] = np.array(likelihoods)
        return ensemble_likelihoods  
    
    ensemble_weights, ensemble_predictions = get_ensemble_weights_and_predictions(split_indices)

    combined_weights = {}
    for key, values in ensemble_predictions.items():
        median_value = custom_median_with_tie(values, threshold=0.5)
        combined_weights[key] = param_transforms[key].inverse(median_value)

    ensemble_likelihoods = get_ensemble_likelihoods(split_indices, combined_weights)

    ensemble_mask = None
    for key, likelihoods in ensemble_likelihoods.items():
        mask = np.all(np.isfinite(likelihoods), axis=0)
        if ensemble_mask is None:
            ensemble_mask = mask
        else:
            ensemble_mask &= mask

    for key, likelihoods in ensemble_likelihoods.items():
        ensemble_likelihoods[key] = likelihoods[:, ensemble_mask]

    y_true = y_true[ensemble_mask]

    combined_likelihoods = {}
    for key, likelihoods in ensemble_likelihoods.items():
        combined_likelihoods[key] = np.mean(likelihoods, axis=0)

    if nbootstrap is None:
        poisson_seeds = None
    else:
        poisson_seeds = [None] + (np.arange(nbootstrap) + seed).tolist()

    scan_kwargs = {
        'mu_arr': mu_arr,
        'alpha_arr': alpha_arr,
        'y_true': y_true,
        'batchsize': scan_batchsize,
        'poisson_seeds': poisson_seeds,
        **combined_likelihoods
    }
    scan_result = run_likelihood_scan(**scan_kwargs)
    
    result = {
        'model_prediction': ensemble_predictions,
        'scan_result': scan_result
    }
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    with open(outname, 'w') as file:
        json.dump(result, file, cls=NpEncoder)
    model_trainer.stdout.info(f"Saved semi-weakly model landscape output to {outname}")

def get_fisher_matrix_old(model, x: np.ndarray, parameters: List[str], batchsize:int =1024):
    from paws.utils import get_parameter_transforms
    from paws.components import ModelLoader
    import tensorflow as tf
    
    param_transforms = get_parameter_transforms(backend='python')

    target_weights = {}
    for index, weight in enumerate(model.trainable_weights):
        name = weight.name.split('/')[0]
        if name not in parameters:
            continue
        target_weights[name] = weight
    sources = list(target_weights.values())
    x = np.asarray(x)
    i_start = 0
    fisher_matrices_batched = []
    while i_start + batchsize  <= len(x):
        x_slice = x[i_start : i_start + batchsize]
        with tf.GradientTape(persistent=True) as tape:
            llr = tf.math.log(model(x_slice))
            score = tape.jacobian(llr, sources)
        score = np.array(score)
        fisher_matrices_batched.append(score[None, :, :] * score[:, None, :])
        i_start += batchsize
        
    fisher_matrices_batched = np.concatenate(fisher_matrices_batched, axis = 2)

    nparams = len(sources)
    return np.sum(fisher_matrices_batched, axis = 2).reshape(nparams, nparams)

def get_fisher_matrix(model, x: np.ndarray, parameters: List[str], batchsize: int = 1024):
    """
    Compute empirical Fisher information matrix at MLE θ*.

    Parameters:
    ----------
    model : tf.keras.Model
        Your trained TensorFlow model, outputting likelihoods per sample.
    x : np.ndarray
        Data samples, shape (num_samples, ...).
    parameters : List[str]
        Names of parameters (layers) whose Fisher matrix you want.
    batchsize : int
        Batch size for computation.

    Returns:
    -------
    fisher_matrix : np.ndarray
        Fisher information matrix, shape (n_params_total, n_params_total).
    """
    import tensorflow as tf
    # Get selected trainable parameters
    target_weights = []
    for weight in model.trainable_weights:
        layer_name = weight.name.split('/')[0]
        if layer_name in parameters:
            target_weights.append(weight)

    num_params = np.sum([tf.size(w).numpy() for w in target_weights])
    fisher_matrix = np.zeros((num_params, num_params), dtype=np.float64)

    # Helper to flatten gradients
    def flatten_gradients(grads):
        return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

    dataset = tf.data.Dataset.from_tensor_slices(x).batch(batchsize)

    for batch_x in dataset:
        with tf.GradientTape() as tape:
            tape.watch(target_weights)
            ll = tf.math.log(model(batch_x))  # shape (batchsize,)
        
        # Compute Jacobian: shape [(batchsize, ...param_shape...), ...]
        grads = tape.jacobian(ll, target_weights)

        batch_size_actual = batch_x.shape[0]
        grads_flat = []
        for i in range(batch_size_actual):
            # Extract and flatten gradients for the i-th sample
            sample_grads = [g[i] for g in grads]
            grads_flat.append(flatten_gradients(sample_grads).numpy())

        grads_flat = np.stack(grads_flat, axis=0)  # shape (batchsize, num_params)

        # Fisher matrix update (batch-wise sum of outer products)
        fisher_matrix += grads_flat.T @ grads_flat

    return fisher_matrix

@cli.command(name='compute_fisher_information')
@click.option('-p', '--parameters', default=None, type=str, show_default=True,
              help='Parameters involved in Fisher matrix calculation (separated by commas). Default to all parameters.')
@click.option('-m', '--mass-point', required=True,
              help='Signal mass point (in the form m1:m2) to use for creating the dataset.')
@click.option('--mu', required=True, type=float,
              help='Signal fraction used in the dataset.')
@click.option('--alpha', default=0.5, type=float, show_default=True,
              help='Signal branching fraction in the dataset. Ignored '
             'when only one signal decay mode is considered.')
@click.option('--kappa', default='1.0', type=str, show_default=True,
              help='Prior normalization factor. It can be a number (fixing kappa value), or a string '
              '. If string, it should be either "sampled" (kappa learned from sampling) or '
              '"inferred" (kappa learned from event number).')
@click.option('--custom-mle', default=None, type=str, show_default=True,
              help='Custom values for parameters at MLE.')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False), show_default=True,
              help='Which decay mode should the signal undergo (qq or qqq).'
              'Use "qq,qqq" to include both decay modes.')
@click.option('--use-trained-weight/--use-true-weight', default=False, show_default=True,
              help='Whether to use the trained model to initialize the weight.')
@click.option('--variables', default=None, show_default=True,
              help='Select certain high-level jet features to include in the training'
              'by the indices they appear in the feature vector. For example,'
              '"3,5,6" means select the 4th, 6th and 7th feature from the jet'
              'feature vector to be used in the training.')
@click.option('--signal', default='W_qq,W_qqq',
              show_default=True,
              help='Signal samples to be used in weakly dataset mixing.')
@click.option('--data-background', default='QCD,extra_QCD',
              show_default=True,
              help='Data background samples to be used in weakly dataset mixing.')
@click.option('--reference-background', default='QCD,extra_QCD',
              show_default=True,
              help='Reference background samples to be used in weakly dataset mixing.')
@click.option('--noise', 'noise_dimension', default=0, type=int, show_default=True,
              help='Number of noise dimension to add to the train features.')
@click.option('--train-data-size', default=None, type=int, show_default=True,
          help='Restrict the number of background events as data (label = 1) used in training.')
@click.option('--val-data-size', default=None, type=int, show_default=True,
          help='Restrict the number of background events as data (label = 1) used in validation.')
@click.option('--test-data-size', default=None, type=int, show_default=True,
          help='Restrict the number of background events as data (label = 1) used in testing.')
@click.option('--dataset-index-path', 'index_path', default=None, show_default=True,
              help='\b\n Path to the dataset split configuration file. It determines the'
              '\b\n shard indices for the train, validation, and test datasets in each'
              '\b\n random realization of data. If None, a default configuration '
              '\b\n will be created.')
@click.option('-i', '--split-index', default=0, type=int, show_default=True,
              help='Index for dataset split.')
@click.option('--fs-split-index', default=None, type=int, show_default=True,
              help='Index for prior model dataset split. Use -1 to include prior '
              'models from all dataset splits. If None, the same dataset split index '
              'as the semi-weakly model will be used.')
@click.option('--ds-type', default='test', type=click.Choice(['train', 'val', 'test'], case_sensitive=False), show_default=True,
          help='Type of dataset to use for evaluation (choose between train, val and test).')
@click.option('--version', default="v1", show_default=True,
              help='Version of the model.')
@click.option('--fs-version', 'fs_version', default="v1", show_default=True,
              help='Version of the supervised model to use.')
@click.option('--fs-version-2', 'fs_version_2', default=None, show_default=True,
              help='\b\n When signals of mixed decay modes are considered, it corresponds to '
             '\b\n the version of the three-prone supervised model. If None, the '
             '\b\n same version as `fs_version` will be used.')
@click.option('--tag', default='default', show_default=True,
              help='Extra tag added to the output directory tree.')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='The default seed used for all random processes.')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for the dataset.')
@click.option('--fisher-batchsize', default=1024, type=int, show_default=True,
              help='Batch size for the fisher matrix computation.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset during training.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='base output directory')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the results.')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU computation.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def compute_fisher_information(**kwargs):
    from quickstats.parsers import ParamParser
    from quickstats.utils.common_utils import NpEncoder, list_of_dict_to_dict_of_list
    from quickstats.utils.string_utils import split_str
    from paws.components import MetricLandscape, ModelLoader
    from paws.utils import get_parameter_transforms

    fisher_parameters = kwargs.pop('parameters')
    custom_mle = kwargs.pop('custom_mle')

    ds_type = kwargs.pop('ds_type')
    use_trained_weight = kwargs.pop('use_trained_weight')
    fisher_batchsize = kwargs.pop('fisher_batchsize')
    tag = kwargs.pop('tag')

    kwargs['loss'] = 'nll'
    kwargs['weakly_test'] = True
    kwargs['use_regularizer'] = False
    model_trainer = get_model_trainer("semi_weakly", **kwargs)

    mixed_signal = len(model_trainer.model_loader.decay_modes) > 1
    
    parameters = model_trainer.model_loader._get_param_repr()
    model_options = model_trainer.model_options
    parameters['mass_point'] = model_options["mass_point"]
    parameters['mu'] = model_options["mu"]
    parameters['alpha'] = model_options["alpha"]
    parameters['split_index'] = kwargs['split_index']
    parameters = model_trainer.path_manager.process_parameters(**parameters)
    parameters['tag'] = tag
    outname = model_trainer.path_manager.get_file(
        "semi_weakly_fisher",
        **parameters,
        ds_type=ds_type
    )
    
    if kwargs['cache'] and os.path.exists(outname):
        model_trainer.stdout.info(f"Cached semi-weakly model landscape output from {outname}")
        return
    
    datasets = model_trainer.get_datasets()

    x = np.concatenate([d[0][0] for d in datasets[ds_type]])
    y_true = np.concatenate([d[1] for d in datasets[ds_type]]).flatten()

    ws_model = model_trainer.get_model()

    param_transforms = get_parameter_transforms(backend='python')
    if custom_mle is not None:
        mle_values = list(ParamParser.parse_param_str(custom_mle))
        if len(mle_values) != 1:
            raise ValueError(
                f'Invalid parameter value specification for MLE: {custom_mle}'
            )
        weights = {}
        for key, value in mle_values[0].items():
            weights[key] = param_transforms[key].inverse(value)
    else:
        if use_trained_weight:
            ws_model_trained = model_trainer.load_trained_model()
            weights = ModelLoader.get_semi_weakly_model_weights(ws_model_trained)
        else:
            mass_point = model_options["mass_point"]
            weights = {
                'm1': param_transforms['m1'].inverse(mass_point[0]),
                'm2': param_transforms['m2'].inverse(mass_point[1]),
                'mu': param_transforms['mu'].inverse(kwargs['mu'])
            }
            if mixed_signal:
                weights['alpha'] = param_transforms['alpha'].inverse(kwargs['alpha'])

    model_prediction = {}
    for key, value in weights.items():
        model_prediction[key] = param_transforms[key](value)
        
    ModelLoader.set_model_weights(ws_model, weights)

    if fisher_parameters is not None:
        fisher_parameters = split_str(fisher_parameters, sep=',', remove_empty=True)
    else:
        fisher_parameters = list(weights)

    fisher_matrix = get_fisher_matrix(ws_model, x, parameters=fisher_parameters, batchsize=fisher_batchsize)

    result = {
        'model_prediction': model_prediction,
        'parameters': fisher_parameters,
        'fisher_matrix': fisher_matrix
    }
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    with open(outname, 'w') as file:
        json.dump(result, file, cls=NpEncoder)
    model_trainer.stdout.info(f"Saved semi-weakly model fisher matrix output to {outname}")

@cli.command(name='gather_model_results')
@click.option('-t', '--model-type', required=True,
              help='Type of model for which the results are gathered.')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq,qqq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False),
              show_default=True,
              help='\b\n Which decay mode should the signal undergo (qq or qqq).'
              '\b\n Use "qq,qqq" to include both decay modes.')
@click.option('--variables', default=None, show_default=True,
              help='\b\n Select certain high-level jet features to include in the training'
              '\b\n by the indices they appear in the feature vector. For example,'
              '\b\n "3,5,6" means select the 4th, 6th and 7th feature from the jet'
              '\b\n feature vector to be used in the training.')
@click.option('-m', '--mass-points', default="*:*", show_default=True, cls=DelimitedStr,
              help='\b\n Filter results by the list of signal mass points in the form m1:m2 '
              '\b\n (separated by commas, wildcard is accepted).')
@click.option('--split-indices', default="*", show_default=True, cls=DelimitedStr,
              help='\b\n Filter results by the list of dataset split indices '
              '\b\n (separated by commas, wildcard is accepted).')
@click.option('--mu-list', default="*", show_default=True, cls=DelimitedStr,
              help='\b\n Filter results by the list of signal fractions. '
              '\b\n (separated by commas, wildcard is accepted).')
@click.option('--alpha-list', default="*", show_default=True, cls=DelimitedStr,
              help='\b\n ilter results by the list of branching fractions. '
              '\b\n (separated by commas, wildcard is accepted).')
@click.option('--trial-list', default="*", show_default=True, cls=DelimitedStr,
              help='Filter results by the list of trial numbers. '
              '(separated by commas, wildcard is accepted).')
@click.option('--noise-list', default="*", show_default=True, cls=DelimitedStr,
              help='\b\n Filter results by the noise dimensions. '
              '\b\n (separated by commas, wildcard is accepted).')
@click.option('--version', default="v1", show_default=True,
              help='Version of the models.')
@click.option('--topk', default=5, show_default=True,
              help='(Weakly models only) Take the top-k trials with lowest loss.')
@click.option('--score-reduce-method', default='mean', type=click.Choice(["mean", "median"]), show_default=True,
              help='(Weakly models only) How to reduce the score over the trials.')
@click.option('--weight-reduce-method', default='median', type=click.Choice(["mean", "median"]), show_default=True,
              help='\b\n (Semi-weakly model only) How to reduce the model weights '
              '\b\n (predicted parameters) over the trials.')
@click.option('--metrics', default="auc,log_loss,sic_1e3", show_default=True,
              cls=DelimitedStr, type=click.Choice(["auc", "accuracy", "log_loss", "sic_1e3", "sic_1e4", "sic_1e5"]), 
              help='\b\n List of metrics to be included in the evaluation (separated by commas). '
              '\b\n Here sic_* refers to the Significance Improvement Characteristic at a 1 / FPR value of *.')
@click.option('--detailed/--simplified', default=False, show_default=True,
              help='Whether to save also the truth and predicted y values of the model results.')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='Base output directory from which model results are extracted.')
@click.option('-f', '--filename', default="{model_type}_{feature_level}_{decay_mode}.parquet", required=True, show_default=True,
              help='\b\n Output filename where the gathered results are saved (on top of outdir). Keywords like '
              '\b\n model_type, feature_level and decay_mode will be automatically formatted.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def gather_model_results(**kwargs):
    """
    Gather model results.
    """
    from quickstats import stdout
    from quickstats.utils.string_utils import split_str
    from paws.components import ResultLoader
    init_kwargs = {}
    for key in ["decay_modes", "variables", "outdir", "verbosity"]:
        init_kwargs[key] = kwargs.pop(key)
    merge_kwargs = {}
    for key in ['topk', 'score_reduce_method', 'weight_reduce_method']:
        merge_kwargs[key] = kwargs.pop(key)        
    feature_level = "high_level" if kwargs.pop("high_level") else "low_level"
    metrics, detailed, filename = [kwargs.pop(key) for key in ["metrics", "detailed", "filename"]]
    result_loader = ResultLoader(feature_level=feature_level, **init_kwargs)
    kwargs["mass_points"] = [split_str(mass_point, sep=":") for mass_point in kwargs["mass_points"]]
    result_loader.load(**kwargs)
    model_type = ModelType.parse(kwargs["model_type"])
    if model_type.key not in result_loader.dfs:
        stdout.warning("No results to save. Skipping.")
        return
    if model_type in [ModelType.SEMI_WEAKLY, ModelType.IDEAL_WEAKLY]:
        result_loader.merge_trials(**merge_kwargs)
    def get_metric(name:str):
        if name.startswith('sic_'):
            threshold = 1 / float(name.replace("sic_", ""))
            return (name, 'threshold_significance', {"fpr_thres": threshold})
        return name
    metrics = [get_metric(metric) for metric in metrics]
    result_loader.decorate_results(metrics)
    format_keys = result_loader._get_param_repr()
    format_keys['model_type'] = model_type.key
    result_loader.path_manager.makedirs(["combined_result"])
    save_outdir = result_loader.path_manager.get_directory("combined_result")
    outname = os.path.join(save_outdir, filename.format(**format_keys))
    result_loader.save_parquet(outname, detailed=detailed)