from typing import Optional, Dict, List, Union, Callable, Tuple
import os
import glob
import json
from itertools import product

import numpy as np
import pandas as pd

from quickstats import semistaticmethod
from quickstats.utils.string_utils import parse_format_str_with_regex
from quickstats.maths.numerics import str_decode_value
from quickstats.extensions import ExtensionDataFrame
from aliad.components.outputs import ModelOutput

from paws.settings import (
    FeatureLevel, ModelType, DecayMode,
    DEDICATED_SUPERVISED, PARAM_SUPERVISED, IDEAL_WEAKLY, SEMI_WEAKLY,
    DEFAULT_FEATURE_LEVEL, DEFAULT_DECAY_MODE, DEFAULT_OUTDIR,
    FEILDS_REGEX, CHECKPOINT_DIR_FMTS, SEMI_WEAKLY_PARAMETERS
)
from paws.utils import (
    get_parameter_transforms
)
from .base_loader import BaseLoader

ArrayType = np.ndarray

class ResultLoader(BaseLoader):

    """
    Class for managing model results.
    """    

    @property
    def dfs(self):
        return self._dfs

    def __init__(
        self,
        feature_level: str = DEFAULT_FEATURE_LEVEL,
        decay_modes: List[str] = DEFAULT_DECAY_MODE,
        variables: Optional[str] = None,
        outdir: str = DEFAULT_OUTDIR,
        verbosity: str = 'INFO',
        **kwargs
    ):
        """
        Initialize the ResultLoader class.
        
        Parameters
        ----------------------------------------------------
        feature_level : str or FeatureLevel, default "high_level"
            Features used for the training. It can be either
            high-level ("high_level") or low-level ("low_level").
        decay_modes : str, list of str or list of DecayMode, default "qq,qqq"
            Decay modes of the signal included in the training. Candidates are
            two-prong decay ("qq") or three-prong decay ("qqq"). If it is a
            string, it will be a comma delimited list of the decay modes.
        variables : str, optional
            Subset of high-level jet features included in the training
            by the indices they appear in the feature vector. For example,
            "3,5,6" means select the 4th, 6th and 7th feature from the jet
            feature vector to be used in the training.
        outdir : str, default "outputs"
            The base path to all outputs (training, plots, etc).
        verbosity : str, default "INFO"
            Verbosity level.
        """
        super().__init__(feature_level=feature_level,
                         decay_modes=decay_modes,
                         variables=variables,
                         outdir=outdir,
                         verbosity=verbosity,
                         **kwargs)
        self.param_transforms = get_parameter_transforms(backend='python')
        self.reset()

    def reset(self) -> None:
        self._dfs = {}

    def get_index_columns(self, columns: List[str]) -> List[str]:
        default_columns = [
            'model_type', 'feature_level', 'decay_mode', 'variables',
            'noise_dim', 'version', 'm1', 'm2', 'split_index', 'mu', 'alpha', 'trial'
        ]
        return [col for col in columns if col in default_columns]

    def get_output_status(
        self,
        model_type: ModelType,
        mass_points: Optional[List[List[float]]] = None,
        split_indices: Optional[List[int]] = None,
        mu_list: Optional[List[float]] = None,
        alpha_list: Optional[List[float]] = None,
        trial_list: Optional[List[int]] = None,
        noise_list: Optional[List[int]] = None,
        version: str = "v1"
    ) -> pd.DataFrame:
        model_type = ModelType.parse(model_type)
        
        def get_default_list(value, shape=(1,)):
            return np.full(shape, fill_value="*") if value is None else value
        
        mass_points = get_default_list(mass_points, shape=(1, 2))
        split_indices = get_default_list(split_indices)
        mu_list = get_default_list(mu_list)
        alpha_list = np.array([None]) if len(self.decay_modes) == 1 else get_default_list(alpha_list)
        trial_list = get_default_list(trial_list)
        noise_list = get_default_list(noise_list)

        keys = ['version', 'mass_point', 'noise_dim', 'split_index', 'mu', 'alpha', 'trial']
        param_values = list(product([version], mass_points, noise_list, split_indices, mu_list, alpha_list, trial_list))
        param_points = [dict(zip(keys, values)) for values in param_values]

        base_params = self._get_param_repr()
        base_params.pop("noise_dim", None)
        dirname = f'{model_type.key}_result'
        checkpoint_dirs = []

        #self.stdout.info(f"Parameter points: {param_points}")
        
        for param_point in param_points:
            parameters = self.path_manager.process_parameters(**param_point, **base_params)
            checkpoint_dir = self.path_manager.get_directory(dirname, **parameters)
            if '*' in checkpoint_dir:
                checkpoint_dirs.extend(glob.glob(checkpoint_dir))
            else:
                checkpoint_dirs.append(checkpoint_dir)

        results = parse_format_str_with_regex(checkpoint_dirs, CHECKPOINT_DIR_FMTS[model_type], FEILDS_REGEX, mode='search')
        data = []
        result_file_basename = self.path_manager.get_file('test_result', basename_only=True)
        for checkpoint_dir, groupdict in results:
            groupdict['checkpoint_dir'] = checkpoint_dir
            groupdict['test_path'] = os.path.join(checkpoint_dir, result_file_basename)
            data.append(groupdict)

        df = pd.DataFrame(data).dropna(axis=1, how='all')
        if df.empty:
            self.stdout.warning(f'No inputs matching the given condition for the {model_type.key} model.')
            return None

        # TODO BETTER WAY TO CHECK TASK STATE
        df['done'] = df['test_path'].apply(os.path.exists)
        df = df.drop(columns=['test_path'])
        
        index_cols = self.get_index_columns(df.columns)
        df = df.drop_duplicates(index_cols, keep='last').reset_index(drop=True)
        return df

    def _fix_dtypes(self, df) -> None:
        for col in ['mu', 'alpha']:
            if col in df.columns:
                df.loc[:, col] = df[col].apply(str_decode_value)
        for col, dtype in [('m1', int), ('m2', int), ('split_index', int), ('trial', int), ('noise_dim', int)]:
            if col in df.columns:
                df.loc[:, col] = df[col].astype(dtype)

    def load_train_summary(
        self,
        checkpoint_dir,
        include_parameters: bool = True,
        include_metrics: bool = True,
        include_monitor: bool = True
    ) -> Dict:
        path = os.path.join(checkpoint_dir, self.path_manager.get_basename("train_summary"))
        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: {path}')
        with open(path) as file:
            data = json.load(file)

        result = {}
        
        if include_metrics and ('best_metrics' in data):
            result.update(data['best_metrics'])

        if include_monitor and ('monitor_metric' in data):
            result['monitor_metric'] = data['monitor_metric']
            result['monitor_mode'] = data['monitor_mode']
        
        if include_parameters and ('predicted_parameters' in data):
            for key, value in data['predicted_parameters'].items():
                result[f'{key}_pred'] = self.param_transforms[key](value)

        return result

    def load_test_file(self, checkpoint_dir:str, do_fix: bool = False):
        filename = os.path.join(checkpoint_dir, self.path_manager.get_basename("test_result"))
        if not os.path.exists(filename):
            raise FileNotFoundError(f'File not found: {filename}')
        with open(filename) as file:
            result = json.load(file)
        if do_fix and ('predicted_proba' in result):
            result['y_pred'] = result.pop('predicted_proba')
        else:
            return result
        backup_filename = os.path.splitext(filename)[0] + '_backup.json'
        os.system(f"mv {filename} {backup_filename}")
        with open(filename, 'w') as file:
            json.dump(result, file)        
        return result
        
    def load(
        self,
        model_type: Union[str, ModelType],
        mass_points: Optional[List[List[float]]] = None,
        split_indices: Optional[List[int]] = None,
        mu_list: Optional[List[float]] = None,
        alpha_list: Optional[List[float]] = None,
        trial_list: Optional[List[int]] = None,
        noise_list: Optional[List[int]] = None,
        version: str = "v1",
        include_output: bool = True,
        include_parameters: bool = True,
        include_metrics: bool = False,
        include_monitor: bool = False,
        likelihood:bool=False,
        update: bool = True,
        do_fix: bool = False,
    ) -> pd.DataFrame:
        """
        Load results for a specific model type.

        Parameters
        ----------------------------------------------------
        model_type : str or ModelType
            The type of model.
        mass_points : List of [float, float], optional
            Filter results by the list of mass points.
        split_indices : list of int, optional
            Filter results by the list of dataset split indices.
        mu_list : list of float, optional
            Filter results by the list of signal fractions.
        alpha_list : list of float, optional
            Filter results by the list of branching fractions.
        trial_list : list of int, optional
            Filter results by the list of trial numbers.
        noise_list : list of int, optional
            Filter results by the noise dimensions.
        version : str
            Version string of the results to look for.
        include_parameters : bool
            Whether to include the parameter prediction for semi-weakly model
        update : str
            Whether to update or overwrite existing results
        """
        model_type = ModelType.parse(model_type)
        if model_type == PARAM_SUPERVISED and mass_points is None:
            raise ValueError('Mass points must be provided when loading parametric supervised model results')

        df = self.get_output_status(model_type,
                                    mass_points=mass_points,
                                    split_indices=split_indices,
                                    mu_list=mu_list,
                                    alpha_list=alpha_list,
                                    trial_list=trial_list,
                                    noise_list=noise_list,
                                    version=version)
        if df is None:
            return None

        df = df[df['done']].drop(columns=['done'])

        if df is None:
            return None
            
        records = df.to_dict('records')
        results = []

        for record in records:
            checkpoint_dir = record['checkpoint_dir']
            self.stdout.info(f'Reading model outputs from "{checkpoint_dir}"')
            
            result = self.load_test_file(checkpoint_dir, do_fix=do_fix)
            result_df = pd.DataFrame(result)

            train_summary = self.load_train_summary(
                checkpoint_dir,
                include_parameters=include_parameters and (model_type == SEMI_WEAKLY),
                include_metrics=include_metrics,
                include_monitor=include_monitor
            )
            record.update(train_summary)

            if include_output:
                if model_type == PARAM_SUPERVISED:
                    for mass_point in mass_points:
                        record_i = record.copy()
                        m1, m2 = mass_point
                        result_df_i = result_df.query(f'((m1 == {m1}) & (m2 == {m2}))')
                        if result_df_i.empty:
                            raise RuntimeError(f'No result found for the mass point (m1, m2) = ({m1}, {m2}) GeV')
                        record_i.update({'m1': m1, 'm2': m2})
                        record_i['output'] = ModelOutput(y_true=result_df_i['y_true'],
                                                         y_pred=result_df_i['y_pred'],
                                                         verbosity=self.stdout.verbosity)
                        results.append(record_i)
                else:
                    if (model_type == SEMI_WEAKLY) and likelihood:
                        if not include_parameters:
                            raise RuntimeError('`include_param` must be true when `likelihood` is True and `model_type` is semi weakly')
                        if 'mu_pred' not in record:
                            raise RuntimeError('`mu_pred` not found in predicted parameters')
                        mu_pred = record['mu_pred']
                        y_pred = np.asarray(result_df['y_pred'])
                        y_prob = y_pred / (y_pred + 1 - mu_pred)
                        record['output'] = ModelOutput(y_true=result_df['y_true'],
                                                       y_pred=y_prob,
                                                       verbosity=self.stdout.verbosity)
                    else:
                        record['output'] = ModelOutput(y_true=result_df['y_true'],
                                                       y_pred=result_df['y_pred'],
                                                       verbosity=self.stdout.verbosity)
                    results.append(record)
            else:
                results.append(record)
        
        result_df = pd.DataFrame(results)
        index_cols = self.get_index_columns(result_df.columns)
        result_df = result_df.sort_values(index_cols).reset_index(drop=True)
        self._fix_dtypes(result_df)
        key = model_type.key
        if (key in self._dfs) and update:
            index_cols = [col for col in result_df.columns if col != 'output']
            self._dfs[key] = pd.concat([self._dfs[key], result_df]).drop_duplicates(index_cols, keep='last').reset_index(drop=True)
        else:
            self._dfs[key] = result_df
        return result_df

    def save_parquet(self, filename: str, detailed: bool = True) -> None:
        """
        Save model results as a parquet file.

        Parameters
        ----------------------------------------------------
        filename : str
            Output filename.
        detailed : bool, default True
            Whether to save also the truth and predicted y values of the model results.
        """
        if not self.dfs:
            raise RuntimeError("No results to save")
        all_records = []
        for model_type, df in self.dfs.items():
            records = df.to_dict('records')
            for record in records:
                record['model_type'] = model_type
                output = record.pop('output', None)
                if (not detailed) or (output is None):
                    continue
                record.update({'y_true': output.data['y_true'], 'y_pred': output.data['y_pred']})
            all_records.extend(records)
        df = pd.DataFrame(all_records)
        df.to_parquet(filename)
        self.stdout.info(f"Saved model results to {filename}")
    
    def load_parquet(self, filename: str, update: bool = True) -> None:
        """
        Load model results from a parquet file.

        Parameters
        ----------------------------------------------------
        filename : str
            Input filename.
        update : bool, default True
            Whether to update or overwrite existing results.
        """        
        df = pd.read_parquet(filename)
        
        def _parse(row):
            return ModelOutput(y_true=row['y_true'], y_pred=row['y_pred'])

        if 'y_true' in df.columns:
            df['output'] = df.apply(_parse, axis=1)
            df = df.drop(columns=['y_true', 'y_pred'])
        
        model_types = df['model_type'].unique()
        for model_type in model_types:
            df_i = df[df['model_type'] == model_type].dropna(axis=1, how='all')
            index_cols = self.get_index_columns(df_i.columns)
            if update and (model_type in self.dfs):
                df_final = pd.concat([self._dfs[model_type], df_i])
            else:
                df_final = df_i
            df_final = df_final.drop_duplicates(index_cols, keep='last').reset_index(drop=True)
            self._dfs[model_type] = df_final

    def _get_reduce_fn(self, reduce_method: str = "mean") -> Callable:
        if reduce_method == 'mean':
            return np.mean
        elif reduce_method == 'median':
            return np.median
        else:
            raise ValueError(f'Unknown reduce method: "{reduce_method}"')

    def _get_nom_and_error(self, values:ArrayType, reduce_method: str = "mean") -> Tuple[ArrayType, ArrayType, ArrayType]:
        if reduce_method == 'mean':
            nominal = np.mean(values)
            std = np.std(values)
            return nominal, nominal - std, nominal + std
        elif reduce_method == 'median':
            nominal = np.median(values)
            return nominal, np.quantile(values, 0.16), np.quantile(values, 0.84)
        else:
            raise ValueError(f'Unknown reduce method: "{reduce_method}"')

    def merge_trials(self, topk: Optional[int] = None,
                     model_types: Optional[List[ModelType]] = None,
                     score_reduce_method: str = "mean",
                     weight_reduce_method: str = "median",
                     mass_ordering: bool = False) -> None:
        """
        Merge weakly and semi-weakly results over initialization trials.
        
        Parameters
        ----------------------------------------------------
        topk : int, optional
            Take results from the top-k trials with the lowest loss.
        model_types : list of str, optional
            The type of model. If None, all existing models will be included.
        score_reduce_method : str, default "mean"
            Method to obtain the merged scores. It can either be "mean"
            (average of the scores) or "median" (median of the scores).
        weight_reduce_method : str, default "median"
            Method to obtain the merged predicetd weights (for semi-weakly models 
            only). It can either be "mean"
            (average of the weights) or "median" (median of the weights).
        mass_ordering : bool, default False
            Whether to order the predicted mass from the semi-weakly model such
            that m1 >= m2.
        """
        if model_types is None:
            model_types = list(self.dfs)
        for model_type in model_types:
            if model_type not in [SEMI_WEAKLY.key, IDEAL_WEAKLY.key]:
                continue
            df = self._get_dataframe(model_type)
            df = df[~df['trial'].isna()].copy()
            if 'output' not in df.columns:
                raise RuntimeError(f'missing y_true and y_pred information from the '
                                   f'results of the model: {model_type}')
            score_reduce = self._get_reduce_fn(score_reduce_method)
            weight_reduce = self._get_reduce_fn(weight_reduce_method)
            weight_columns = ['m1_pred', 'm2_pred', 'mu_pred', 'alpha_pred']
            non_index_columns = weight_columns + ['trial', 'output']
            columns = [c for c in df.columns if c not in non_index_columns]

            merged_results = []
            for values, df_group in df.groupby(columns, dropna=False):
                record = dict(zip(columns, values))
                record['trial'] = np.nan
                outputs = df_group['output'].values
                y_trues = np.array([output.data['y_true'] for output in outputs])
                y_preds = np.array([output.data['y_pred'] for output in outputs])            

                if topk is not None:
                    logloss = np.array([output.log_loss() for output in outputs])
                    indices = np.argsort(logloss)[:topk]
                    y_trues = y_trues[indices]
                    y_preds = y_preds[indices]
                else:
                    indices = None

                if not (y_trues == y_trues[0]).all():
                    raise RuntimeError('y_true not the same across trials, please check your input')

                y_true = y_trues[0]
                y_pred = score_reduce(y_preds, axis=0)
                output = ModelOutput(y_true=y_true, y_pred=y_pred,
                                     verbosity=self.stdout.verbosity)
                record['output'] = output

                if model_type == SEMI_WEAKLY.key:
                    if mass_ordering and record['m1'] != record['m2']:
                        m1_arr = df_group['m1_pred'].values
                        m2_arr = df_group['m2_pred'].values
                        m_arr = np.sort(np.array([m1_arr, m2_arr]), axis=0)
                        df_group.loc[:, 'm1_pred'] = m_arr[1]
                        df_group.loc[:, 'm2_pred'] = m_arr[0]

                    for weight_column in weight_columns:
                        if weight_column in df_group.columns:
                            if indices is not None:
                                weight_values = df_group[weight_column].values[indices]
                            else:
                                weight_values = df_group[weight_column].values
                            record[weight_column] = weight_reduce(weight_values)
                
                merged_results.append(record)

            df_merged = pd.DataFrame(merged_results)
            df_merged['trial'] = df_merged['trial'].astype('Int64')
            df = pd.concat([df, df_merged]).reset_index(drop=True)
            self.dfs[model_type] = df

    MetricsOptionsType = List[Union[str, Tuple[str, str, Dict]]]
    
    def decorate_results(self, metric_options:MetricsOptionsType, model_types: List[str] = None, merged_only: bool = True) -> None:
        """
        Decorate existing model results with the given metrics.

        Parameters
        ----------------------------------------------------
        metric_options : list of (str or tuple of (str, str, dict))
            List of specification of the metrics to be included. If the
            specification is a str, it is the name of the metrics, e.g.
            "auc", "accuracy", "log_loss". Otherwise it should be a tuple
            of (str, str, dict) specifying denoting the custom metric name,
            the name of the metric method, and the keyword arguments to the
            metric method, respectively. For example, a value of
            ("sig_1e3", "threshold_significance", {"fpr_thres": 1e-3})
            corresponds to the significance improvement at a fpr threshold
            of 1e-3.
        model_types : list of str, optional
            The type of model. If None, all existing models will be included.
        merge_only : bool, default True
            Whether to evaluate the metrics for merged (over the trials)
            results only. This is only applicable for weakly and semi-weakly
            model results.
        """
        if model_types is None:
            model_types = list(self.dfs)
        for model_type in model_types:
            df = self._get_dataframe(model_type)
            if (model_type in [SEMI_WEAKLY.key, IDEAL_WEAKLY.key]) and merged_only:
                mask = df['trial'].isna()
            else:
                mask = np.ones(len(df), dtype=bool)
                
            for metric_option in metric_options:
                if isinstance(metric_option, str):
                    metric = metric_option
                    df.loc[mask, metric] = df.loc[mask, 'output'].apply(lambda x: getattr(x, metric)())
                elif isinstance(metric_option, (list, tuple)):
                    colname, metric, options = metric_option
                    df.loc[mask, colname] = df.loc[mask, 'output'].apply(lambda x: getattr(x, metric)(**options))
                else:
                    raise ValueError(f'Invalid metric option format: "{metric_option}"')

    def get_ensemble_result(self, metrics: List[str], model_types: List[str] = None,
                            reduce_method: str = "median", extend_supervised: bool = True,
                            select: Optional[Dict] = None, reject: Optional[Dict] = None,
                            indexed: bool = True, groupby: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Compute the expected values of the metrics and their errors over the dataset initializations (split indices).

        Parameters
        ----------------------------------------------------
        metrics : list of str
            List of metrics names for which the ensembled results are evaluated.
        model_types : list of str, optional
            The type of model. If None, all existing models will be included.
        reduce_method : str, default "median"
            How to evaluate the expected results and the corresponding errors. It can either be
            "median", which uses the median and one-sigma quantiles for the expected results
            and the errors; or "mean", which uses the mean and standard deviation for the
            expected results and the errors.
        extend_supervised : bool, default True
            Whether to replicate the supervised model results over the signal injection (mu) values.
        select : dict
            Specification for selecting the results by the indices. These indices can be feature_level,
            decay_mode, version and so on.
        reject : dict
            Specification for excluding the results by the indices. These indices can be feature_level,
            decay_mode, version and so on.
        indexed : bool
            Whether to index columns that are not part of the metrics / predicted parameters.
        groupby : str, optional
            Convert the results to a dictionary of dataframes with the key being the given column.

        Returns
        ----------------------------------------------------
        pandas.DataFrame or dict of {str : pandas.DataFrame}
            Dataframe (or dictionary of dataframes with the given groupby columns as keys) containing the
            ensemble results.
        """
        if model_types is None:
            model_types = list(self.dfs)
        results = []

        for model_type in model_types:
            df = self._get_dataframe(model_type)
            if model_type in [SEMI_WEAKLY.key, IDEAL_WEAKLY.key]:
                df = df[df['trial'].isna()].drop(columns=['trial'])

            if df.empty:
                self.stdout.warning(f"No results for {model_type} model matching the given condition. Skipped")
                continue

            columns = self.get_index_columns(df.columns)
            columns.remove('split_index')

            for values, df_group in df.groupby(columns, dropna=False):
                record = dict(zip(columns, values))
                record['model_type'] = model_type

                for metric in metrics:
                    if metric not in df_group.columns:
                        raise RuntimeError(f'Metric "{metric}" not initialized for {model_type} model')
                    metric_values = df_group[metric].values
                    nom, errlo, errhi = self._get_nom_and_error(metric_values, reduce_method)
                    record.update({metric: nom, f'{metric}_errlo': errlo, f'{metric}_errhi': errhi})
                
                results.append(record)

        if not results:
            return None

        df = pd.DataFrame(results)
        
        if select:
            df = ExtensionDataFrame(df).select_values(columns=select)
        if reject:
            df = ExtensionDataFrame(df).reject_values(columns=reject)

        if extend_supervised:
            param_keys, param_vals = [], []

            for key in ['mu', 'alpha']:
                if key not in df.columns:
                    continue
                values = df[key].dropna().unique()
                if len(values) == 0:
                    continue
                param_keys.append(key)
                param_vals.append(values)
            combinations = list(product(*param_vals))
            param_points = [dict(zip(param_keys, values)) for values in combinations]
            helper = ExtensionDataFrame(df)
            
            for model_type in [DEDICATED_SUPERVISED.key, PARAM_SUPERVISED.key]:
                df_model = helper.select_values(columns={"model_type": model_type})
                if df_model.empty:
                    continue
                helper.reject_values(columns={"model_type": model_type}, inplace=True)
                extended_dfs = [df_model.assign(**param_point) for param_point in param_points]
                helper.concat(extended_dfs, inplace=True, order='first')

        df = helper.dataframe.dropna(axis=1, how='all')
        
        if groupby:
            groupvals = df[groupby].unique()
            dfs = {}
            for groupval in groupvals:
                df_group = df[df[groupby] == groupval]
                if indexed:
                    df_group = self._make_indexed(df_group)
                dfs[groupval] = df_group
            return dfs
        if indexed:
            df = self._make_indexed(df)
        
        return df

    def _make_indexed(self, df:pd.DataFrame)  -> pd.DataFrame:
        columns = self.get_index_columns(df.columns)
        return df.sort_values(columns).set_index(columns)

    def _get_dataframe(self, model_type:str) -> pd.DataFrame:
        if model_type not in self.dfs:
            raise RuntimeError(f'Results for {model_type} models not initialized')
        return self.dfs[model_type]
        
    def get_dataframe(self, model_type: str,
                      select: Optional[Dict] = None,
                      reject: Optional[Dict] = None) -> pd.DataFrame:
        """
        Get the dataframe representation of the model results.
        
        Parameters
        ----------------------------------------------------
        model_type : list of str, optional
            The type of model.
        select : dict
            Specification for selecting the results by the indices. These indices can be feature_level,
            decay_mode, version and so on.
        reject : dict
            Specification for excluding the results by the indices. These indices can be feature_level,
            decay_mode, version and so on.
        """
        df = self._get_dataframe(model_type)
        if select:
            df = ExtensionDataFrame(df).select_values(columns=select)
        if reject:
            df = ExtensionDataFrame(df).reject_values(columns=reject)
        return df