from typing import Optional, List, Union
import os

from quickstats import AbstractObject
from quickstats.utils.string_utils import split_str

from paws import PathManager
from paws.settings import FeatureLevel, DecayMode, ModelType, PARAM_SUPERVISED
from paws.settings import TRAIN_FEATURES, PARAM_FEATURE
from paws.settings import BASE_SEED, DEFAULT_FEATURE_LEVEL, DEFAULT_DECAY_MODE, DEFAULT_DATADIR, DEFAULT_OUTDIR

class BaseLoader(AbstractObject):
    """
    Base class for all the loader methods.
    """
    
    @property
    def feature_level(self) -> FeatureLevel:
        return self._feature_level

    @feature_level.setter
    def feature_level(self, value: Union[str, FeatureLevel]):
        self._feature_level = FeatureLevel.parse(value)

    @property
    def decay_modes(self) -> List[DecayMode]:
        return self._decay_modes

    @decay_modes.setter
    def decay_modes(self, values: Union[str, List[Union[str, DecayMode]]]):
        if isinstance(values, str):
            values = split_str(values, sep=",", strip=True, remove_empty=True)
        decay_modes = [DecayMode.parse(v) for v in values]
        if len(decay_modes) not in [1, 2]:
            raise ValueError('Number of decay modes must be either 1 or 2')
        self._decay_modes = sorted(decay_modes, key=lambda x: x.value)

    @property
    def variables(self) -> Optional[List[str]]:
        return self._variables

    @variables.setter
    def variables(self, value: Optional[str]):
        if value:
            self._variables = split_str(value, sep=",", strip=True, remove_empty=True)
        else:
            self._variables = None
            
    @property
    def noise_dimension(self) -> int:
        return self._noise_dimension

    @noise_dimension.setter
    def noise_dimension(self, value: Optional[int]):
        if not value:
            value = 0
        if not isinstance(value, int):
            raise TypeError(f'noise dimension must be an integer')
        if not value % 2 == 0:
            raise ValueError(f'noise dimension must be divisible by the number of jets')
        self._noise_dimension = value

    @property
    def noise_dimension_per_jet(self) -> int:
        return self.noise_dimension // 2

    def __init__(self, feature_level: Union[str, FeatureLevel] = DEFAULT_FEATURE_LEVEL,
                 decay_modes: Union[str, List[Union[str, DecayMode]]] = DEFAULT_DECAY_MODE,
                 variables: Optional[str] = None,
                 noise_dimension: Optional[int] = None,
                 use_validation: bool = True,
                 seed: Optional[int] = BASE_SEED,
                 datadir: str = DEFAULT_DATADIR,
                 outdir: str = DEFAULT_OUTDIR,
                 distribute_strategy = None,
                 verbosity: str = 'INFO'):
        """
        Initialize the BaseLoader class.
        
        Parameters
        ----------------------------------------------------
        feature_level : str or FeatureLevel, default "high_level"
            Features to use for the training. It can be either
            high-level ("high_level") or low-level ("low_level").
        decay_modes : str, list of str or list of DecayMode, default "qq,qqq"
            Decay modes of the signal to include in the training. Candidates are
            two-prong decay ("qq") or three-prong decay ("qqq"). If it is a
            string, it will be a comma delimited list of the decay modes.
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
        outdir : str, default "outputs"
            The base path to all outputs (training, plots, etc).
        distribute_strategy : tf.distribute.Strategy
            Strategy used for distributed (multi-GPU) training.
        verbosity : str, default "INFO"
            Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").
        """
        super().__init__(verbosity=verbosity)
        self.feature_level = feature_level
        self.decay_modes = decay_modes
        self.variables = variables
        self.noise_dimension = noise_dimension
        self.use_validation = use_validation
        self.seed = seed
        self.distribute_strategy = distribute_strategy
        self.path_manager = PathManager()
        self.path_manager.set_directory("dataset", datadir)
        self.path_manager.set_directory("output", outdir)  

    def _get_param_repr(self) -> str:
        param_repr = {
            'feature_level': self.feature_level.key,
            'decay_mode': self.path_manager._get_decay_mode_repr(self.decay_modes),
            'variables': self.path_manager._get_variables_repr(self.variables),
            'noise_dim': str(self.noise_dimension)
        }
        return param_repr

    def _get_train_features(self, model_type: Union[str, ModelType]) -> List[str]:
        model_type = ModelType.parse(model_type)
        features = list(TRAIN_FEATURES[self.feature_level])
        if model_type == PARAM_SUPERVISED:
            features.append(PARAM_FEATURE)
        return features

    def _get_param_feature(self):
        return PARAM_FEATURE

    def _get_aux_features(self, model_type: Union[str, ModelType]) -> List[str]:
        # TODO: possibility of weighted events
        return ['label']
        
    def _get_required_features(self, model_type: Union[str, ModelType]) -> List[str]:
        train_features = self._get_train_features(model_type=model_type)
        aux_features = self._get_aux_features(model_type=model_type)
        return train_features + aux_features

    def get_checkpoint_dir(self, model_type: Union[str, ModelType],
                           mass_point:Optional[List[float]]=None,
                           mu:Optional[float]=None,
                           alpha:Optional[float]=None,
                           split_index:Optional[int]=None,
                           trial:Optional[int]=None,
                           version:Optional[str]=None,
                           **kwargs):
        model_type = ModelType.parse(model_type)
        dirname = f'{model_type.key}_result'
        parameters = self._get_param_repr()
        parameters.update({
            'mass_point': mass_point,
            'mu': mu,
            'alpha': alpha,
            'split_index': split_index,
            'trial': trial,
            'version': version
        })
        parameters.update(kwargs)
        parameters = self.path_manager.process_parameters(**parameters)
        return self.path_manager.get_directory(dirname, **parameters)

    def get_checkpoint_path(self, filename:str, checkpoint_dir:str):
        return os.path.join(checkpoint_dir, self.path_manager.get_basename(filename))