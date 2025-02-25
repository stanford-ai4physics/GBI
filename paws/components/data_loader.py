from typing import Optional, Dict, List, Union, Tuple, Callable
from collections import defaultdict
from operator import itemgetter
from functools import cache
import os
import glob
import json

import numpy as np
import pandas as pd

from quickstats.utils.string_utils import split_str
from quickstats.maths.numerics import get_max_sizes_from_fraction
from aliad.data.partition import optimize_split_sizes

from paws.settings import FeatureLevel, HIGH_LEVEL, LOW_LEVEL
from paws.settings import DecayMode, TWO_PRONG, THREE_PRONG
from paws.settings import ModelType, DEDICATED_SUPERVISED, PARAM_SUPERVISED, SEMI_WEAKLY
from paws.settings import Sample, NUM_SHARDS, SPLIT_FRACTIONS, BASE_SEED
from paws.settings import DEFAULT_FEATURE_LEVEL, DEFAULT_DECAY_MODE, DEFAULT_DATADIR

from .base_loader import BaseLoader

class DataLoader(BaseLoader):
    """
    Class for managing dataset splits and transformations.
    """
    
    def __init__(
        self,
        feature_level: Union[str, FeatureLevel] = DEFAULT_FEATURE_LEVEL,
        decay_modes: Union[str, List[Union[str, DecayMode]]] = DEFAULT_DECAY_MODE,
        index_path: Optional[str] = None,
        variables: Optional[str] = None,
        noise_dimension: Optional[int] = None,
        seed: Optional[int] = BASE_SEED,
        datadir: str = DEFAULT_DATADIR,
        distribute_strategy = None,
        verbosity: str = 'INFO',
        **kwargs
    ):
        """
        Initialize the DataLoader class.
        
        Parameters
        ----------------------------------------------------
        feature_level : str or FeatureLevel, default "high_level"
            Features to use for the training. It can be either
            high-level ("high_level") or low-level ("low_level").
        decay_modes : str, list of str or list of DecayMode, default "qq,qqq"
            Decay modes of the signal to include in the training. Candidates are
            two-prong decay ("qq") or three-prong decay ("qqq"). If it is a
            string, it will be a comma delimited list of the decay modes.
        index_path : str, optional
            Path to the dataset split configuration file. It determines the
            shard indices for the train, validation, and test datasets in each
            random realization of data.
        variables : str, optional
            Select certain high-level jet features to include in the training
            by the indices they appear in the feature vector. For example,
            "3,5,6" means select the 4th, 6th and 7th feature from the jet
            feature vector to be used in the training.
        noise_dimension : int, optional
            Number of noise dimension to include in the training. It must be
            divisible by the number of jets (i.e. 2).
        seed : int, optional, default 2023
            The default seed used for all random processes.
        datadir : str, default "datasets"
            The base path to the input datasets.
        distribute_strategy : tf.distribute.Strategy
            Strategy used for distributed (multi-GPU) training.
        verbosity : str, default "INFO"
            Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").     
        """
        super().__init__(feature_level=feature_level,
                         decay_modes=decay_modes,
                         variables=variables,
                         noise_dimension=noise_dimension,
                         seed=seed,
                         datadir=datadir,
                         distribute_strategy=distribute_strategy,
                         verbosity=verbosity,
                         **kwargs)
        self.set_split_config(index_path)
        self.feature_metadata = None
        self.dataset_summary = None

    def generate_split_config(
        self,
        split_fractions: str = SPLIT_FRACTIONS,
        num_shards: int = NUM_SHARDS,
        num_splits: int = 100
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Generate a configuration for dataset splits. The split config has the form
        {<split_index>: {"train": <shard_indices>, "val": <shard_indices>, "test": <shard_indices>}, ...}

        Here each `split_index` specifies a unique configuration for splitting the dataset. The `shard_indices`
        are the indices of the tfrecord dataset shards used in a particular stage, i.e. training (train),
        validation (val) and testing (test).

        Parameters
        ----------------------------------------------------
        split_fractions : str
            Split fractions as a colon-separated string. E.g. 50:25:25 means
            a 50-25-25 splits into train, validation and test datasets.
        num_shards : int
            Number of shards.
        num_splits : int
            Number of splits.

        Returns
        ----------------------------------------------------
        dict of {int : dict of {str : np.ndarry}}
            The split configuration.
        """
        split_sizes = split_str(split_fractions, sep=':', cast=lambda x: float(x) / 100.)
        train_size, val_size, test_size = optimize_split_sizes(num_shards, split_sizes=split_sizes)
        
        config = {}
        np.random.seed(self.seed)
        for i in range(num_splits):
            indices = np.arange(num_shards) if i == 0 else np.random.permutation(num_shards)
            config[i] = {
                'train': indices[:train_size],
                'val': indices[train_size: train_size + val_size],
                'test': indices[train_size + val_size: train_size + val_size + test_size]
            }
        return config

    def get_complimentary_split_config(
        self,
        split_config: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Generate a complimentary split configuration with orthogonal train, val and test definitions.

        Parameters
        ----------------------------------------------------
        split_config : dict of {int : dict of {str : np.ndarray}}
            Original split configuration.

        Returns
        ----------------------------------------------------
        dict of {int : dict of {str : np.ndarry}}
            The complimentary split configuration.
        """
        split_config_comp = {}

        @cache
        def compute_comp_split_sizes(train_size: int, val_size: int, test_size: int) -> Tuple[int, int, int]:
            c_train_size = val_size + test_size
            val_frac = val_size / c_train_size
            test_frac = test_size / c_train_size
            c_val_size, c_test_size = optimize_split_sizes(train_size, split_sizes=[val_frac, test_frac])
            return c_train_size, c_val_size, c_test_size
            
        for i, splits in split_config.items():
            _, val_size, test_size = compute_comp_split_sizes(len(splits['train']), len(splits['val']), len(splits['test']))
            split_config_comp[i] = {
                'train': np.concatenate([splits['val'], splits['test']]),
                'val': splits['train'][:val_size],
                'test': splits['train'][val_size: val_size + test_size]
            }
        return split_config_comp

    def set_split_config(self, index_path: Optional[str] = None):
        if index_path is None:
            config = self.generate_split_config()
        else:
            with open(index_path, 'r') as file:
                config = json.load(file)
                config = {int(k): v for k, v in config.items()}
        config_comp = self.get_complimentary_split_config(config)
        self.split_config = config
        self.split_config_comp = config_comp

    def _suggest_batchsize(self, batchsize: Optional[int] = None) -> int:
        if batchsize is None:
            return 1024 if self.feature_level == HIGH_LEVEL else 128
        return batchsize

    def _suggest_cache_dataset(self, cache_dataset: Optional[bool] = None) -> bool:
        if cache_dataset is None:
            return self.feature_level == HIGH_LEVEL
        return cache_dataset

    def _get_filter_fn(self, param_values: List[float], mode: str = "include") -> Callable:
        import tensorflow as tf
        
        param_tensor = tf.constant(param_values, dtype='float64')
        
        param_feature = self._get_param_feature()
        if mode == "exclude":
            return lambda x: tf.reduce_all(tf.reduce_any(tf.not_equal(x[param_feature], param_tensor), axis=1))
        if mode == "include":
            return lambda x: tf.reduce_any(tf.reduce_all(tf.equal(x[param_feature], param_tensor), axis=1))
        raise ValueError(f"Unknown filter mode: {mode}")

    def _get_all_filters(self, include_params: Optional[List[List[float]]] = None,
                         exclude_params: Optional[List[List[float]]] = None) -> List[Callable]:
        filters = []
        if include_params:
            filters.append(self._get_filter_fn(include_params, mode='include'))
        if exclude_params:
            filters.append(self._get_filter_fn(exclude_params, mode='exclude'))
        return filters

    def _get_custom_label_transform(self, label: Union[int, Tuple[int]] = None) -> Callable:
        import tensorflow as tf
        if isinstance(label, (int, float)):
            label = [label]
        value = tf.constant(label, dtype='int64')
        def modify_label(X, value=value):
            X['label'] = value
            return X
        return modify_label

    def _get_custom_param_transform(self, param_values: Union[int, Tuple[int]]) -> Callable:
        import tensorflow as tf
        param_feature = self._get_param_feature()
        value = tf.constant(param_values, dtype='float64')
        def modify_param_values(X, value=value):
            X[param_feature] = value
            return X
        return modify_param_values

    def _get_custom_indices_transform(self, indices: List[int], feature: str = 'jet_features') -> Callable:
        import tensorflow as tf
        var_index = tf.constant([int(i) for i in indices])
        def select_indices(X):
            X[feature] = tf.gather(X[feature], var_index, axis=1)
            return X
        return select_indices

    def _get_noise_transform(self, noise_dimension: int, feature: str = 'jet_features') -> Callable:
        import tensorflow as tf
        def add_noise_dimension(X):
            noise = tf.random.normal((tf.shape(X[feature])[0], noise_dimension), dtype=X[feature].dtype)
            X[feature] = tf.concat([X[feature], noise], -1)
            return X
        return add_noise_dimension

    def _get_all_transforms(
        self,
        model_type: ModelType,
        custom_params: Optional[List[float]] = None,
        custom_label: Optional[Union[int, Tuple[int]]] = None
    ) -> List[Callable]:
        from aliad.interface.tensorflow.dataset import feature_selector
        
        transforms = []
        
        if custom_label is not None:
            transforms.append(self._get_custom_label_transform(custom_label))
            
        if custom_params is not None:
            transforms.append(self._get_custom_param_transform(custom_params))
            
        if self.variables is not None:
            transforms.append(self._get_custom_indices_transform(self.variables))

        if self.noise_dimension_per_jet:
            transforms.append(self._get_noise_transform(self.noise_dimension_per_jet))

        model_type = ModelType.parse(model_type)
        train_features = self._get_train_features(model_type)
        if custom_params is not None:
            param_feature = self._get_param_feature()
            train_features.append(param_feature)
        aux_features = self._get_aux_features(model_type)
        input_fn = feature_selector(train_features, aux_features)
        transforms.append(input_fn)
        
        return transforms

    def _resolve_samples(
        self,
        samples: Optional[List[str]] = None
    ) -> List[Sample]:
        if samples is None:
            return [sample for sample in Sample if sample.default_sample and \
                    ((sample.decay_mode is None) or (sample.decay_mode in self.decay_modes))]
        return [Sample.parse(sample) for sample in samples]

    def _resolve_label_map(
        self,
        label_map: Optional[Dict[str, int]] = None
    ) -> Dict[Sample, int]:
        if label_map is None:
            return {}
        resolved_label_map = {}
        for sample, label in label_map.items():
            sample = Sample.parse(sample)
            resolved_label_map[sample] = label
        return resolved_label_map
        
    def get_dataset_specs(
        self,
        mass_point: Optional[List[float]] = None,
        samples: Optional[List[str]] = None,
        supervised_label_map: Optional[Dict[str, int]] = None,
        weakly_label_map: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """
        Generate dataset specifications for the given mass point and samples.

        Parameters
        ----------------------------------------------------
        mass_point: list of floats, optional
            List containing mass points [m1, m2].
        samples: list of strings, optional
            List of sample names.

        Returns
        ----------------------------------------------------
        pandas.DataFrame
            DataFrame containing dataset specifications.
        """
        parameters = self._get_param_repr()
        
        if mass_point is not None:
            dirname = "dedicated_dataset"
            m1, m2 = mass_point
            parameters.update(m1=int(m1), m2=int(m2))
            samples = self._resolve_samples(samples)
        else:
            dirname = "param_dataset"
            samples = [Sample.MIXED]
            
        specs = defaultdict(list)

        supervised_label_map = self._resolve_label_map(supervised_label_map)
        weakly_label_map = self._resolve_label_map(weakly_label_map)

        get_shard_index = lambda f: int(os.path.splitext(f)[0].split("_")[-1])
        for sample in samples:
            sample_dir = self.path_manager.get_directory(dirname, sample=sample.key, **parameters)
            if not os.path.exists(sample_dir):
                raise FileNotFoundError(f"Sample directory does not exist: {sample_dir}")
                
            dataset_paths = glob.glob(os.path.join(sample_dir, "*.tfrecord"))
            if not dataset_paths:
                raise RuntimeError(f"No dataset files found for the sample '{sample.key}' under the directory '{sample_dir}'")
            dataset_paths = sorted(dataset_paths, key=get_shard_index)
            specs['sample'].extend([sample.key] * len(dataset_paths))
            specs['dataset_path'].extend(dataset_paths)
            
            if sample != Sample.MIXED:
                supervised_label = supervised_label_map.get(sample, sample.supervised_label)
                specs['supervised_label'].extend([supervised_label] * len(dataset_paths))
                weakly_label = weakly_label_map.get(sample, sample.weakly_label)
                specs['weakly_label'].extend([weakly_label] * len(dataset_paths))
                decay_mode = Sample.parse(sample).decay_mode
                specs['decay_mode'].extend([decay_mode] * len(dataset_paths))

        df = pd.DataFrame(specs)
        if df.empty:
            raise RuntimeError('Generated dataset specs is empty. Check your dataset paths and samples.')
            
        df['metadata_path'] = df['dataset_path'].apply(lambda f: f'{os.path.splitext(f)[0]}_metadata.json')
        df['shard_index'] = df['dataset_path'].apply(get_shard_index)
        
        def get_dataset_size(f):
            with open(f) as metadata_file:
                metadata = json.load(metadata_file)
                return metadata['size']
        
        df['size'] = df['metadata_path'].apply(get_dataset_size)

        return df

    def _get_sample_composition(
        self,
        df: pd.DataFrame,
        mu: Optional[float] = None,
        alpha: Optional[float] = None
    ) -> List[Dict]:
        composition = []
        # supervised dataset
        if mu is None:
            for sample in df['sample'].unique():
                sample_df = df[df['sample'] == sample]
                sample_size = sample_df['size'].sum()
                components = {
                    'name': sample,
                    'skip': 0,
                    'take': sample_size,
                    'size': sample_size                    
                }
                if 'supervised_label' in sample_df.columns:
                    components['label'] = sample_df['supervised_label'].values[0]
                composition.append({
                    'paths': sample_df['dataset_path'].values,
                    'components': [components]
                })
            return composition
        if 'supervised_label' not in df.columns:
            raise ValueError('Cannot specify signal fraction (mu) for already mixed dataset')
        masks = {
            'bkg': df['supervised_label'] == 0,
            'sig': df['supervised_label'] == 1,
            'two_prong': df['decay_mode'] == TWO_PRONG,
            'three_prong': df['decay_mode'] == THREE_PRONG
        }
        masks['ref_bkg'] = masks['bkg'] & (df['weakly_label'] == 0)
        masks['data_bkg'] = masks['bkg'] & (df['weakly_label'] == 1)
        masks['data_sig'] = masks['sig'] & (df['weakly_label'] == 1)
        
        sizes = {sample_type: int(df[mask]['size'].sum()) for sample_type, mask in masks.items()}
        # sanity checks
        if sizes['sig'] == 0 and mu != 0.:
            raise ValueError('No signal samples specified but requested signal fraction > 0')
        if sizes['bkg'] == 0 and mu != 1.:
            raise ValueError('No background samples specified but requested signal fraction < 1')
        if alpha is None and sizes['two_prong'] > 0 and sizes['three_prong'] > 0:
            raise ValueError('Branching fraction not specified but samples from multiple decay modes are found in the dataset')
        if alpha is not None and (sizes['two_prong'] == 0 or sizes['three_prong'] == 0):
            raise ValueError('Dataset must include both two-prong and three-prong signals when branching fraction is specified')
        if alpha is not None:
            two_prong_size, three_prong_size = get_max_sizes_from_fraction(sizes['two_prong'],
                                                                           sizes['three_prong'],
                                                                           1 - alpha)
            sig_size_init = two_prong_size + three_prong_size
        else:
            two_prong_size = three_prong_size = None
            sig_size_init = sizes['sig']
        if (sizes['ref_bkg'] == 0) and (sizes['data_bkg'] == 0):
            ref_size = data_size_init = int(sizes['bkg'] / 2)
        elif (sizes['ref_bkg'] > 0) and (sizes['data_bkg'] > 0):
            ref_size = data_size_init = min(sizes['ref_bkg'], sizes['data_bkg'])
        elif (sizes['ref_bkg'] == 0) and (sizes['data_bkg'] > 0):
            ref_size = 0
            data_size_init = sizes['data_bkg']
        else:
            raise ValueError('At least one background samples must have weakly label 1')

        if sizes['sig'] != sizes['data_sig']:
            raise ValueError('Invalid weakly label for signal samples (must be 1)')
        
        sig_size, data_size = get_max_sizes_from_fraction(sig_size_init, data_size_init, mu)
        # signal is limiting
        if (ref_size != 0) and (data_size < ref_size):
            required_sig_size = int((data_size_init + sig_size_init) * mu)
            self.stdout.warning(f"Number of available signal events ({sig_size_init}) not "
                                f"enough to compose dataset with a signal fraction of {mu} "
                                f"(requires {required_sig_size} events). Will shrink size of "
                                f"background events from {data_size_init} to {data_size}", "red")
            ref_size = data_size
        # background is limiting
        elif alpha is not None and sig_size < sig_size_init:
            two_prong_size = int(sig_size * (1 - alpha))
            three_prong_size = int(sig_size * alpha)

        def get_key(sample_df):
            return ' + '.join(sample_df['sample'].unique())

        if (sizes['ref_bkg'] == 0) and (sizes['data_bkg'] == 0):
            bkg_df = df[masks['bkg']]
            bkg_key = get_key(bkg_df)
            composition.append({
                'paths': bkg_df['dataset_path'].values,
                'components': [
                    {'name': f'{bkg_key} (label = 0)', 'label': 0, 'skip': 0, 'take': ref_size, 'size': sizes['bkg']},
                    {'name': f'{bkg_key} (label = 1)', 'label': 1, 'skip': ref_size, 'take': data_size, 'size': sizes['bkg']}
                ]
            })
        else:
            for key, take_size in [('data_bkg', data_size), ('ref_bkg', ref_size)]:
                if take_size == 0:
                    continue
                bkg_df = df[masks[key]]
                bkg_key = get_key(bkg_df)
                label = bkg_df['weakly_label'].values[0]
                composition.append({
                    'paths': bkg_df['dataset_path'].values,
                    'components': [
                        {'name': f'{bkg_key} (label = {label})', 'label': label, 'skip': 0, 'take': take_size, 'size': sizes[key]}
                    ]
                })
                
        if alpha is None:
            size_mask = [(sig_size, masks['sig'])]
        else:
            size_mask = [(two_prong_size, masks['two_prong']), (three_prong_size, masks['three_prong'])]
        for size, mask in size_mask:
            sample_df = df[mask]
            sample_size = sample_df['size'].sum()
            composition.append({
                'paths': sample_df['dataset_path'].values,
                'components': [{'name': get_key(sample_df), 'label': 1, 'skip': 0, 'take': size, 'size': sample_size}]
            })

        return composition

    def _get_sample_size_summary(self, sample_composition: List[Dict], batchsize: Optional[int] = None) -> Dict[str, int]:
        summary = {}
        total_size = 0
        for composition in sample_composition:
            for component in composition['components']:
                sample_name = component['name']
                sample_size = int(component['take'])
                summary[sample_name] = sample_size
                total_size += sample_size
        summary['total'] = total_size
        if batchsize is not None:
            summary['num_batch'] = summary['total'] // batchsize
            if (summary['total'] % batchsize) != 0:
                summary['num_batch'] += 1
        return summary

    def _get_dataset_from_sample_composition(
        self,
        sample_composition: List[Dict],
        parse_tfrecord_fn: Callable,
        transforms: Optional[List[Callable]] = None,
        filters: Optional[List[Callable]] = None
    ):
        import tensorflow as tf
        from aliad.interface.tensorflow.dataset import get_tfrecord_dataset, concatenate_datasets

        if transforms is None:
            transforms = []
        if filters is None:
            filters = []

        label_transforms = {}

        ds_list = []
        for composition in sample_composition:
            ds_0 = get_tfrecord_dataset(composition['paths'], parse_tfrecord_fn)
            for component in composition['components']:
                ds_i = ds_0
                if not component['take']:
                    continue
                if component['skip']:
                    ds_i = ds_i.skip(component['skip'])
                if component['take'] != component['size']:
                    ds_i = ds_i.take(component['take'])
                for filter_fn in filters:
                    ds_i = ds_i.filter(filter_fn)
                if 'label' in component:
                    label = component['label']
                    if label not in label_transforms:
                        label_transforms[label] = self._get_custom_label_transform(label)
                    ds_i = ds_i.map(label_transforms[label], num_parallel_calls=tf.data.AUTOTUNE)
                for transform in transforms:
                    ds_i = ds_i.map(transform, num_parallel_calls=tf.data.AUTOTUNE)                   
                ds_list.append(ds_i)
        ds = concatenate_datasets(ds_list)
        return ds

    def _print_dataset_summary(self, summary: Dict[str, Dict[str, int]]):
        df = pd.DataFrame(summary).fillna(0).astype(int)
        if 'mixed' in df.index:
            df = df.drop(index=['mixed'])
        def move_to_end(df, indices):
            rows = df.loc[indices]
            df = df.drop(indices)
            return pd.concat([df, rows])
        df = move_to_end(df, ['total', 'num_batch'])
        self.stdout.info('Number of events in each dataset split:')
        self.stdout.info(df, bare=True)

    def get_datasets(
        self,
        mass_point: Optional[List[float]] = None,
        mu: Optional[float] = None,
        alpha: Optional[float] = None,
        samples: Optional[List[str]] = None,
        supervised_label_map: Optional[Dict[str, int]] = None,
        weakly_label_map: Optional[Dict[str, int]] = None,
        custom_masses: Optional[List[float]] = None,
        include_masses: Optional[List[List[float]]] = None,
        exclude_masses: Optional[List[List[float]]] = None,                                
        split_index: int = 0,
        batchsize: Optional[int] = None,
        cache_dataset: Optional[bool] = None,
        cache_test: bool = False
    ) -> Dict[str, "tf.data.Dataset"]:
        """
        Get datasets for training, validation, and testing.

        If `mass_point` is None, a parameterised dataset will be created.
        Otherwise, a dedicated dataset for the given mass point will
        be created.

        If `mu` is None, a supervised dataset will be created. Otherwise,
        a weakly dataset with the given `mu` value as the signal fraction
        will be created.

        Parameters
        ----------------------------------------------------
        mass_point : list of float, optional
            Signal masses in the form (m1, m2).
        mu : float, optional
            Signal fraction.
        alpha : float, optional
            Branching fraction.
        samples : list of str, optional
            (dedicated dataset) List of samples to include.
        custom_masses : list of float, optional
            (dedicated dataset) Specify custom values for the mass parameters
            in the dataset.
        include_masses : list of list of float, optional
            (parameterised dataset) Mass points to include.
        exclude_masses : list of list of float, optional
            (parameterised dataset) Mass points to exclude.
        split_index : int, default 0
            Index for split configuration.
        batchsize : int, optional
            Batch size. If None, it will be automatically deduced from the feature level.
        cache_dataset : bool, optional
            Whether to cache the dataset. If None, it will be automatically deduced from the feature level.
        cache_test : bool, default False
            Whether to cache the test dataset.

        Returns
        ----------------------------------------------------
        dict of {str : tf.data.Dataset}
            Dictionary containing datasets for each stage.
        """
        from aliad.interface.tensorflow.dataset import get_tfrecord_array_parser, apply_pipelines

        if mass_point is None:
            model_type = PARAM_SUPERVISED
            if samples is not None:
                raise ValueError('Cannot specify samples for mixed dataset')
            if mu is not None or alpha is not None:
                raise ValueError('Cannot specify mu or alpha for mixed dataset')
        else:
            model_type = DEDICATED_SUPERVISED if mu is None else SEMI_WEAKLY
            if include_masses is not None or exclude_masses is not None:
                raise ValueError('Masses selection not allowed for dedicated dataset')
            if model_type == SEMI_WEAKLY and custom_masses:
                raise ValueError('Cannot specify custom masses for weakly dataset')
                
        spec_df = self.get_dataset_specs(mass_point, samples=samples,
                                         supervised_label_map=supervised_label_map,
                                         weakly_label_map=weakly_label_map)
        
        with open(spec_df['metadata_path'].iloc[0], 'r') as file:
            metadata = json.load(file)
        self.feature_metadata = metadata['features']
        
        required_features = self._get_required_features(model_type)
        
        parse_tfrecord_fn = get_tfrecord_array_parser(metadata['features'], keys=required_features)
        
        filters = self._get_all_filters(include_params=include_masses,
                                        exclude_params=exclude_masses)
        transforms = self._get_all_transforms(model_type, custom_params=custom_masses)

        if model_type == SEMI_WEAKLY:
            split_config = self.split_config_comp[split_index]
        else:
            split_config = self.split_config[split_index]
        batchsize = self._suggest_batchsize(batchsize)
        cache_dataset = self._suggest_cache_dataset(cache_dataset)

        import tensorflow as tf
        
        all_ds, summary = {}, {}
        
        for stage in split_config:

            shuffle = (model_type != PARAM_SUPERVISED) and stage == 'train'
            cache = cache_dataset and (stage != 'test' or cache_test)
            distribute_strategy = self.distribute_strategy if stage != 'test' else None
            
            stage_mask = spec_df['shard_index'].isin(split_config[stage])
            stage_spec_df = spec_df[stage_mask]

            stage_mu = mu if model_type == SEMI_WEAKLY and stage != 'test' else None
            sample_composition = self._get_sample_composition(stage_spec_df, mu=stage_mu, alpha=alpha)
            summary[stage] = self._get_sample_size_summary(sample_composition, batchsize)
            ds = self._get_dataset_from_sample_composition(sample_composition, parse_tfrecord_fn,
                                                           transforms=transforms, filters=filters)
            buffer_size = summary[stage]['total'] if shuffle else None
            ds = apply_pipelines(ds,
                                 batch_size=batchsize,
                                 cache=cache,
                                 shuffle=shuffle,
                                 seed=self.seed,
                                 prefetch=True,
                                 buffer_size=buffer_size,
                                 drop_remainder=False,
                                 reshuffle_each_iteration=False,
                                 distribute_strategy=distribute_strategy)
            all_ds[stage] = ds
            
        self.dataset_summary = summary
        self._print_dataset_summary(summary)
        return all_ds