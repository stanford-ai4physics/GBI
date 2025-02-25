from typing import Optional, Union, Dict, List, Tuple
import os

from quickstats import PathManager as PathManagerBase
from quickstats.maths.numerics import str_encode_value

from paws.settings import DecayMode, DEDICATED_SUPERVISED, PARAM_SUPERVISED, IDEAL_WEAKLY, SEMI_WEAKLY
from paws.settings import CHECKPOINT_DIR_FMTS

class PathManager(PathManagerBase):
    
    DEFAULT_DIRECTORIES = {
        "dataset"              : "datasets",
        "original_dataset"     : ("dataset", "original"),
        "processed_dataset"    : ("dataset", "processed"),
        "dedicated_dataset"    : ("processed_dataset", "{feature_level}/dedicated/{sample}_{m1}_{m2}"),
        "param_dataset"        : ("processed_dataset", "{feature_level}/parameterised/{decay_mode}"),
        "output"               : "outputs",
        "plot"                 : ("output", "plots"),
        "landscape"            : ("output", "landscapes"),
        "fisher"               : ("output", "fisher_information"),
        "train_result"         : ("output", "train_results"),
        "combined_result"      : ("output", "summary"),
        "dedicated_supervised_result" : ("train_result", CHECKPOINT_DIR_FMTS[DEDICATED_SUPERVISED]),
        "param_supervised_result"     : ("train_result", CHECKPOINT_DIR_FMTS[PARAM_SUPERVISED]),
        "ideal_weakly_result"         : ("train_result", CHECKPOINT_DIR_FMTS[IDEAL_WEAKLY]),
        "semi_weakly_result"          : ("train_result", CHECKPOINT_DIR_FMTS[SEMI_WEAKLY])
    }

    DEFAULT_FILES = {
        "dedicated_dataset"           : ("dedicated_dataset", "SR_train_features_shard_{{shard_index}}.tfrecord"),
        "param_dataset"               : ("param_dataset", "SR_train_features_shard_{{shard_index}}.tfrecord"),
        "train_config"                : ("{model_type}_result", "train_config.json"),
        "dataset_summary"             : ("{model_type}_result", "dataset_summary.json"),
        "model_full_train"            : ("{model_type}_result", "full_train.keras"),
        "model_checkpoint"            : ("{model_type}_result", "model_weights_epoch_{{epoch:02d}}.h5"),
        "metrics_checkpoint"          : ("{model_type}_result", "epoch_metrics/metrics_epoch_{{epoch}}.json"),
        "test_result"                 : ("{model_type}_result", "test_results.json"),
        "predicted_parameters"        : ("{model_type}_result", "predicted_parameters.json"),
        "train_summary"               : ("{model_type}_result", "summary.json"),
        "param_supervised_landscape"  : ("landscape", "param_supervised/{feature_level}/"
                                         "{ds_type}_{split_index}/{decay_mode}/landscape_{m1}_{m2}.json"),
        "semi_weakly_landscape"       : ("landscape", "semi_weakly/{feature_level}/{tag}/{mu_alpha}/"
                                         "{ds_type}_{split_index}/{decay_mode}/landscape_{m1}_{m2}.json"),
        "semi_weakly_fisher"          : ("fisher", "semi_weakly/{feature_level}/{tag}/{mu_alpha}/"
                                         "{ds_type}_{split_index}/{decay_mode}/landscape_{m1}_{m2}.json"),
        "model_prior_ratio"           : ("param_supervised_result", "full_train_prior_ratio_{sampling_method}.keras"),
        "prior_ratio_dataset"         : ("param_supervised_result", "prior_ratio_dataset_{sampling_method}.json"),
    }

    @staticmethod
    def _get_decay_mode_repr(decay_modes:List[DecayMode]):
        return '_'.join([decay_mode.key for decay_mode in decay_modes])

    @staticmethod
    def _get_variables_repr(variables:Optional[List[str]]):
        return '_'.join(variables) if (variables is not None) else 'all'

    @staticmethod
    def _get_mu_alpha_repr(mu:float, alpha:Optional[float]=None):
        try:
            mu_str = str_encode_value(mu)
        except Exception:
            mu_str = mu
        mu_alpha_str = f'mu_{mu_str}'
        if alpha is not None:
            try:
                alpha_str = str_encode_value(alpha)
            except Exception:
                alpha_str = alpha
            mu_alpha_str += f'_alpha_{alpha_str}'
        return mu_alpha_str

    @staticmethod
    def process_parameters(**parameters) -> None:
        parameters = {key: value for key, value in parameters.items() if value is not None}
        if 'mass_point' in parameters:
            m1, m2 = parameters.pop('mass_point')
            parameters['m1'], parameters['m2'] = m1, m2
        if 'mu' in parameters:
            mu, alpha = parameters.pop('mu'), parameters.pop('alpha', None)
            parameters['mu_alpha'] = PathManager._get_mu_alpha_repr(mu, alpha)
        if 'decay_modes' in parameters:
            decay_modes = parameters.pop('decay_modes')
            parameters['decay_mode'] = PathManager._get_decay_mode_repr(decay_modes)
        return parameters