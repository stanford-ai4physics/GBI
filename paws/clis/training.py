from typing import Union
import os

import click

from quickstats.utils.string_utils import split_str

from .main import cli
from paws.settings import DEFAULT_DATADIR, DEFAULT_OUTDIR, BASE_SEED

__all__ = ["train_dedicated_supervised", "train_param_supervised",
           "train_ideal_weakly", "train_semi_weakly"]

kCommonKeys = ["high_level", "decay_modes", "variables", "noise_dimension", "loss",
               "index_path", "split_index", "seed", "batchsize", "interrupt_freq",
               "cache_dataset", "datadir", "outdir", "version", "cache", "multi_gpu",
               "epochs", "verbosity", "use_validation", "model_save_freq",
               "metric_save_freq", "weight_save_freq"]

def train_model(model_type:str, **kwargs):
    from paws.components import ModelTrainer
    init_kwargs = {key: kwargs.pop(key) for key in kCommonKeys if key in kwargs}
    if 'cache_prior_dataset' in kwargs:
        kwargs['cache_dataset'] = kwargs.pop('cache_prior_dataset')

    # dedicated supervised or prior ratio
    if ('signal' in kwargs) and ('background' in kwargs):
        if kwargs['signal'] is None:
            kwargs.pop('signal')
            sig_samples = []
        else:
            sig_samples = list(set(split_str(kwargs.pop('signal'), sep=',', remove_empty=True)))
        if kwargs['background'] is None:
            kwargs.pop('background')
            bkg_samples = []
        else:
            bkg_samples = list(set(split_str(kwargs.pop('background'), sep=',', remove_empty=True)))
        if not set(sig_samples).isdisjoint(set(bkg_samples)):
            raise ValueError('Signal and background samples must be disjoint')
        kwargs['samples'] = sig_samples + bkg_samples
        if model_type.lower() == 'dedicated_supervised':
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

    init_kwargs['model_options'] = kwargs
    feature_level = "high_level" if init_kwargs.pop("high_level") else "low_level"
    model_trainer = ModelTrainer(model_type, **init_kwargs)
    model_trainer.train()
    
@cli.command(name='train_dedicated_supervised')
@click.option('-m', '--mass-point', required=True, show_default=True,
              help='Signal mass point to use for training in the form "m1:m2".')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq,qqq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False),
              show_default=True,
              help='\b\n Which decay mode should the signal undergo (qq or qqq).'
              '\b\n Use "qq,qqq" to include both decay modes.')
@click.option('--signal', default='W_qq,W_qqq',
              show_default=True,
              help='Signal samples to be used in training.')
@click.option('--background', default='QCD,extra_QCD',
              show_default=True,
              help='Background samples to be used in training.')
@click.option('--variables', default=None, show_default=True,
              help='\b\n Select certain high-level jet features to include in the training'
              '\b\n by the indices they appear in the feature vector. For example,'
              '\b\n "3,5,6" means select the 4th, 6th and 7th feature from the jet'
              '\b\n feature vector to be used in the training.')
@click.option('--noise', 'noise_dimension', default=0, type=int, show_default=True,
              help='Number of noise dimension to add to the train features.')
@click.option('--dataset-index-path', 'index_path', default=None, show_default=True,
              help='\b\n Path to the dataset split configuration file. It determines the'
              '\b\n shard indices for the train, validation, and test datasets in each'
              '\b\n random realization of data. If None, a default configuration '
              '\b\n will be created.')
@click.option('-i', '--split-index', default=0, type=int, show_default=True,
              help='Index for dataset split.')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='The default seed used for all random processes.')
@click.option('--epochs', default=None, type=int, show_default=True,
              help='Number of epochs for training.')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for training.')
@click.option('--interrupt-freq', default=None, type=int, show_default=True,
              help='Frequency of training interruption for early stopping.')
@click.option('--model-save-freq', default='train', show_default=True,
              help='Frequency of saving model checkpoints.')
@click.option('--metric-save-freq', default='train', show_default=True,
              help='Frequency of saving train metrics.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset during training.')
@click.option('--use-validation/--no-validation', 'use_validation', default=True, show_default=True,
              help='Whether to use validation data in the training.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='Input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='Base output directory')
@click.option('--version', default="v1", show_default=True,
              help='Version of the model.')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the results.')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU training.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def train_dedicated_supervised(**kwargs):
    """
    Train dedicated supervised models.
    """
    train_model("dedicated_supervised", **kwargs)

@cli.command(name='train_param_supervised')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False), show_default=True,
              help='\b\n Which decay mode should the signal undergo (qq or qqq).'
              '\b\n Use "qq,qqq" to include both decay modes.')
@click.option('--variables', default=None, show_default=True,
              help='\b\n Select certain high-level jet features to include in the training'
              '\b\n by the indices they appear in the feature vector. For example,'
              '\b\n "3,5,6" means select the 4th, 6th and 7th feature from the jet'
              '\b\n feature vector to be used in the training.')
@click.option('--noise', 'noise_dimension', default=None, type=int, show_default=True,
              help='Number of noise dimension to add to the train features.')
@click.option('--exclude-masses', default=None, show_default=True,
              help='Mass points to exclude (mass point separated by commas, mass values separated by colon).')
@click.option('--include-masses', default=None, show_default=True,
              help='Mass points to include (mass point separated by commas, mass values separated by colon).')
@click.option('--dataset-index-path', 'index_path', default=None, show_default=True,
              help='\b\n Path to the dataset split configuration file. It determines the'
              '\b\n shard indices for the train, validation, and test datasets in each'
              '\b\n random realization of data. If None, a default configuration '
              '\b\n will be created.')
@click.option('-i', '--split-index', default=0, type=int, show_default=True,
              help='Index for dataset split.')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='The default seed used for all random processes.')
@click.option('--epochs', default=None, type=int, show_default=True,
              help='Number of epochs for training.')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for training.')
@click.option('--interrupt-freq', default=None, type=int, show_default=True,
              help='Frequency of training interruption for early stopping.')
@click.option('--model-save-freq', default='epoch', show_default=True,
              help='Frequency of saving model checkpoints.')
@click.option('--metric-save-freq', default='epoch', show_default=True,
              help='Frequency of saving train metrics.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset during training.')
@click.option('--use-validation/--no-validation', 'use_validation', default=True, show_default=True,
              help='Whether to use validation data in the training.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='Input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='Base output directory')
@click.option('--version', default="v1", show_default=True,
              help='Version of the model.')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the results.')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU training.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def train_param_supervised(**kwargs):
    """
    Train parameterised supervised models.
    """
    train_model("param_supervised", **kwargs)

@cli.command(name='train_prior_ratio')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False), show_default=True,
              help='\b\n Which decay mode should the signal undergo (qq or qqq).'
              '\b\n Use "qq,qqq" to include both decay modes.')
@click.option('--variables', default=None, show_default=True,
              help='\b\n Select certain high-level jet features to include in the training'
              '\b\n by the indices they appear in the feature vector. For example,'
              '\b\n "3,5,6" means select the 4th, 6th and 7th feature from the jet'
              '\b\n feature vector to be used in the training.')
@click.option('--signal', default=None,
              show_default=True,
              help='Signal samples to be used for prior ratio calculation (for interred method only).')
@click.option('--background', default='QCD,extra_QCD',
              show_default=True,
              help='Background samples to be used for prior ratio calculation.')
@click.option('--noise', 'noise_dimension', default=None, type=int, show_default=True,
              help='Number of noise dimension to add to the train features.')
@click.option('--dataset-index-path', 'index_path', default=None, show_default=True,
              help='\b\n Path to the dataset split configuration file. It determines the'
              '\b\n shard indices for the train, validation, and test datasets in each'
              '\b\n random realization of data. If None, a default configuration '
              '\b\n will be created.')
@click.option('-i', '--split-index', default=0, type=int, show_default=True,
              help='Index for dataset split.')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='The default seed used for all random processes.')
@click.option('--epochs', default=None, type=int, show_default=True,
          help='Number of epochs for training.')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for computing prior dataset.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset used for sampling prior ratio.')
@click.option('--model-save-freq', default='epoch', show_default=True,
              help='Frequency of saving model checkpoints.')
@click.option('--metric-save-freq', default='epoch', show_default=True,
              help='Frequency of saving train metrics.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='Input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='Base output directory')
@click.option('--version', default="v1", show_default=True,
              help='Version of the model.')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the prior model.')
@click.option('--cache-prior-dataset/--no-cache-prior-dataset', default=True, show_default=True,
              help='Whether to cache the prior dataset.')
@click.option('--sampling-method', default='sampled', show_default=True,
              help='How to sample the prior ratio.')
@click.option('--param-expr', default=None, show_default=True,
              help='Expression for generating the set of mass points for prior ratio sampling. '
              'Only used when sampling_method is "sampled"')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU training.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def train_prior_ratio(**kwargs):
    """
    Train prior ratio calibration model.
    """
    train_model("prior_ratio", **kwargs)

@cli.command(name='train_ideal_weakly')
@click.option('-m', '--mass-point', required=True, show_default=True,
              help='Signal mass point to use for training in the form "m1:m2".')
@click.option('--mu', required=True, type=float,
              help='Signal fraction in the training and validation dataset.')
@click.option('--alpha', default=0.5, type=float,
              help='\b\n Signal branching fraction in the training and validation dataset. Ignored '
             '\b\n when only one signal decay mode is considered.')
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
@click.option('--signal', default='W_qq,W_qqq',
              show_default=True,
              help='Signal samples to be used in weakly dataset mixing.')
@click.option('--data-background', default='QCD,extra_QCD',
              show_default=True,
              help='Data background samples to be used in weakly dataset mixing.')
@click.option('--reference-background', default='QCD,extra_QCD',
              show_default=True,
              help='Reference background samples to be used in weakly dataset mixing.')
@click.option('--noise', 'noise_dimension', default=None, type=int, show_default=True,
              help='Number of noise dimension to add to the train features.')
@click.option('--dataset-index-path', 'index_path', default=None, show_default=True,
              help='\b\n Path to the dataset split configuration file. It determines the'
              '\b\n shard indices for the train, validation, and test datasets in each'
              '\b\n random realization of data. If None, a default configuration '
              '\b\n will be created.')
@click.option('-i', '--split-index', default=0, type=int, show_default=True,
              help='Index for dataset split.')
@click.option('--num-trials', default=10, type=int, show_default=True,
              help='Number of trials (random model initialization) to run.')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='The default seed used for all random processes.')
@click.option('--epochs', default=None, type=int, show_default=True,
              help='Number of epochs for training.')
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for training.')
@click.option('--interrupt-freq', default=None, type=int, show_default=True,
              help='Frequency of training interruption for early stopping.')
@click.option('--model-save-freq', default='train', show_default=True,
              help='Frequency of saving model checkpoints.')
@click.option('--metric-save-freq', default='train', show_default=True,
              help='Frequency of saving train metrics.')
@click.option('--train-data-size', default=None, type=int, show_default=True,
              help='Restrict the number of background events as data (label = 1) used in training.')
@click.option('--val-data-size', default=None, type=int, show_default=True,
              help='Restrict the number of background events as data (label = 1) used in validation.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset during training.')
@click.option('--use-validation/--no-validation', 'use_validation', default=True, show_default=True,
          help='Whether to use validation data in the training.')
@click.option('--ref-bkg-test/--no-ref-bkg-test', default=False, show_default=True,
          help='Whether to use reference background in test dataset.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='Input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='Base output directory')
@click.option('--version', default="v1", show_default=True,
              help='Version of the model.')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the results.')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU training.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def train_ideal_weakly(**kwargs):
    """
    Train ideal weakly models.
    """
    train_model("ideal_weakly", **kwargs)

@cli.command(name='train_semi_weakly')
@click.option('-m', '--mass-point', required=True, show_default=True,
              help='Signal mass point to use for training in the form "m1:m2".')
@click.option('--mu', required=True, type=float,
              help='Signal fraction in the training and validation dataset.')
@click.option('--alpha', default=0.5, type=float,
              help='\b\n Signal branching fraction in the training and validation dataset. Ignored '
             '\b\n when only one signal decay mode is considered.')
@click.option('--kappa', default='1.0', type=str,
              help='Prior normalization factor. It can be a number (fixing kappa value), or a string '
              '. If string, it should be either "sampled" (kappa learned from sampling) or '
              '"inferred" (kappa learned from event number).')
@click.option('--high-level/--low-level', default=True, show_default=True,
              help='Whether to do training with low-evel or high-level features.')
@click.option('--decay-modes', default='qq,qqq', type=click.Choice(['qq', 'qqq', 'qq,qqq'], case_sensitive=False),
              show_default=True,
              help='Which decay mode should the signal undergo (qq or qqq).'
              'Use "qq,qqq" to include both decay modes.')
@click.option('--variables', default=None, show_default=True,
              help='\b\n Select certain high-level jet features to include in the training'
              '\b\n by the indices they appear in the feature vector. For example,'
              '\b\n "3,5,6" means select the 4th, 6th and 7th feature from the jet'
              '\b\n feature vector to be used in the training.')
@click.option('--signal', default='W_qq,W_qqq',
              show_default=True,
              help='Signal samples to be used in weakly dataset mixing.')
@click.option('--data-background', default='QCD,extra_QCD',
              show_default=True,
              help='Data background samples to be used in weakly dataset mixing.')
@click.option('--reference-background', default='QCD,extra_QCD',
              show_default=True,
              help='Reference background samples to be used in weakly dataset mixing.')
@click.option('--noise', 'noise_dimension', default=None, type=int, show_default=True,
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
@click.option('--num-trials', default=10, type=int, show_default=True,
              help='Number of trials (random model initialization) to run.')
@click.option('--fs-version', 'fs_version', default="v1", show_default=True,
              help='Version of the supervised model to use.')
@click.option('--fs-version-2', 'fs_version_2', default=None, show_default=True,
              help='\b\n When signals of mixed decay modes are considered, it corresponds to '
             '\b\n the version of the three-prone supervised model. If None, the '
             '\b\n same version as `fs_version` will be used.')
@click.option('--retrain/--no-retrain', default=False, show_default=True,
              help='Retrain when m1 <-> m2 gives better validation loss.')
@click.option('--use-sigmoid/--not-use-sigmoid', default=False, show_default=True,
              help='Whether to use sigmoid activation.')
@click.option('--seed', default=BASE_SEED, type=int, show_default=True,
              help='The default seed used for all random processes.')
@click.option('--epochs', default=None, type=int, show_default=True,
              help='Number of epochs for training.')    
@click.option('--batchsize', default=None, type=int, show_default=True,
              help='Batch size for training.')
@click.option('--interrupt-freq', default=None, type=int, show_default=True,
              help='Frequency of training interruption for early stopping.')
@click.option('--model-save-freq', default='train', show_default=True,
              help='Frequency of saving model checkpoints.')
@click.option('--metric-save-freq', default='train', show_default=True,
              help='Frequency of saving train metrics.')
@click.option('--weight-save-freq', default='train', show_default=True,
              help='Frequency of saving model weights.')    
@click.option('--train-data-size', default=None, type=int, show_default=True,
              help='Restrict the number of background events as data (label = 1) used in training.')
@click.option('--val-data-size', default=None, type=int, show_default=True,
              help='Restrict the number of background events as data (label = 1) used in validation.')
@click.option('--weight-clipping/--no-weight-clipping', default=True, show_default=True,
              help='Whether to apply weight clipping.')
@click.option('--cache-dataset/--no-cache-dataset', default=None, show_default=True,
              help='Whether to cache the dataset during training.')
@click.option('--use-validation/--no-validation', 'use_validation', default=True, show_default=True,
          help='Whether to use validation data in the training.')
@click.option('--ref-bkg-test/--no-ref-bkg-test', default=False, show_default=True,
          help='Whether to use reference background in test dataset.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='Input directory where the tfrecord datasets are stored')
@click.option('-o', '--outdir', default=DEFAULT_OUTDIR, show_default=True,
              help='Base output directory')
@click.option('--version', default="v1", show_default=True,
              help='Version of the model.')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache the results.')
@click.option('--multi-gpu/--single-gpu', default=True, show_default=True,
              help='Whether to enable multi-GPU training.')
@click.option('-v', '--verbosity',  default="INFO", show_default=True,
              help='Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").')
def train_semi_weakly(**kwargs):
    """
    Train semi-weakly (PAWS) models.
    """    
    train_model("semi_weakly", **kwargs)