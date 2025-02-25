import os

import click

from .main import cli, DelimitedStr
from paws.settings import kSampleList, SampleURLs, DEFAULT_DATADIR, NUM_SHARDS, BASE_SEED

__all__ = ['download_data', 'create_dedicated_datasets', 'create_param_datasets']



@cli.command(name='download_data')
@click.option('-s', '--samples', default=",".join(kSampleList),
              cls=DelimitedStr, type=click.Choice(kSampleList),
              help='List of data samples to download (separated by commas). Available samples are: '
              f'{kSampleList}. By default, all samples will be downloaded.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='Base directory for storing datasets. The downloaded data will be stored in <datadir>/raw.')
def download_data(**kwargs):
    """
    Download datasets used in this study.
    """
    from quickstats import stdout
    from paws import PathManager
    from paws.data_preparation import download_file
    from paws.settings import Sample
    path_manager = PathManager(directories={"dataset": kwargs['datadir']})
    outdir = path_manager.get_directory('original_dataset')
    for sample in kwargs['samples']:
        url = SampleURLs[Sample.parse(sample)]
        stdout.info(f'Downloading sample "{sample}" from {url}')
        download_file(url, outdir)

@cli.command(name='create_dedicated_datasets')
@click.option('-s', '--samples', default=",".join(kSampleList),
              cls=DelimitedStr, type=click.Choice(kSampleList),
              help='List of data samples to process (separated by commas). Available samples are: '
              f'{kSampleList}. By default, all samples will be processed.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='Base directory for storing datasets.')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache existing results.')
@click.option('--parallel', type=int, default=-1, show_default=True,
              help='\b\n Parallelize job across the N workers.'
                   '\b\n Case  0: Jobs are run sequentially (for debugging).'
                   '\b\n Case -1: Jobs are run across N_CPU workers.')
@click.option('-v', '--verbosity', default='INFO', show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              help='Verbosity level.')
def create_dedicated_datasets(**kwargs):
    """
    Create dedicated datasets for model training.
    """
    from quickstats import stdout, switch_verbosity
    from paws.data_preparation import create_high_level_dedicated_datasets
    samples = kwargs.pop("samples")
    verbosity = kwargs.pop("verbosity")
    with switch_verbosity(stdout, verbosity):
        for sample in samples:
            create_high_level_dedicated_datasets(sample, **kwargs)

@cli.command(name='create_param_datasets')
@click.option('-s', '--samples', default=",".join(kSampleList),
              cls=DelimitedStr, type=click.Choice(kSampleList),
              help='List of data samples to include (separated by commas). Available samples are: '
              f'{kSampleList}. By default, all samples will be included. Note that for '
              'two-prong / three-prong training, only the two-prong / three-prong signals '
              'should be included.')
@click.option('-d', '--datadir', default=DEFAULT_DATADIR, show_default=True,
              help='Base directory for storing datasets.')
@click.option('--shards', default=None, show_default=True,
              help='Process datasets with the specific shard indices (separated by commas). '
              'By default, all shards will be processed. Use "start_index:end_index" to '
              'indicate a slice of shard indices.')
@click.option('--seed', default=BASE_SEED, show_default=True,
              help='Random seed used in dataset shuffling.')
@click.option('--cache/--no-cache', default=True, show_default=True,
              help='Whether to cache existing results.')
@click.option('--parallel', type=int, default=16, show_default=True,
              help='\b\n Parallelize job across the N workers.'
                   '\b\n Case  0: Jobs are run sequentially (for debugging).'
                   '\b\n Case -1: Jobs are run across N_CPU workers.')
@click.option('-v', '--verbosity', default='INFO', show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              help='Verbosity level.')
def create_param_datasets(**kwargs):
    """
    Create parameterised datasets for model training.
    """
    import numpy as np
    
    from quickstats import stdout, switch_verbosity
    from quickstats.utils.string_utils import split_str
    from paws.data_preparation import create_parameterised_datasets
    
    shards = kwargs.pop('shards')
    verbosity = kwargs.pop('verbosity')
    if shards is None:
        shard_indices = np.arange(NUM_SHARDS)
    else:
        shard_indices = []
        shards = split_str(shards, sep=",")
        for shard in shards:
            if ':' in shard:
                start, end = split_str(shard, sep=":", cast=int)
                shard_indices.extend(np.arange(start, end))
            else:
                shard_indices.append(int(shard))
    with switch_verbosity(stdout, verbosity):
        create_parameterised_datasets(shard_indices, **kwargs)
