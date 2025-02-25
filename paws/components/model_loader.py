from typing import Optional, Dict, List, Union, Tuple, Any
from numbers import Number
import os
import json

import numpy as np

from quickstats import semistaticmethod
from quickstats.core.modules import get_module_version
from quickstats.utils.string_utils import split_str

from paws.settings import (
    FeatureLevel, HIGH_LEVEL, LOW_LEVEL, TRAIN_FEATURES, ModelType, SEMI_WEAKLY, IDEAL_WEAKLY,
    MLP_LAYERS, INIT_MU, INIT_ALPHA, INIT_KAPPA,
    DEFAULT_FEATURE_LEVEL, DEFAULT_DECAY_MODE, DEFAULT_OUTDIR,
    MASS_RANGE, MASS_SCALE, PRIOR_RATIO, PRIOR_RATIO_NET_LAYERS
)
from paws.utils import (
    get_parameter_transform,
    get_parameter_regularizer
)
from .base_loader import BaseLoader

def assign_weight(weight: "tf.Variable", value: Any):
    from aliad.interface.tensorflow import utils
    utils.assign_weight(weight, value)

class ModelLoader(BaseLoader):
    """
    Class for managing the loading and configuration of models.
    """

    def __init__(self, feature_level: str = DEFAULT_FEATURE_LEVEL,
                 decay_modes: List[str] = DEFAULT_DECAY_MODE,
                 variables: Optional[str] = None,
                 noise_dimension: Optional[int] = None,
                 sigmoid_activation: bool = False,
                 loss: str = 'bce',
                 use_validation: bool = True,
                 distribute_strategy = None,
                 outdir: str = DEFAULT_OUTDIR,                 
                 verbosity: str = 'INFO',
                 **kwargs):
        """
        Initialize the ModelLoader class.
        
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
        loss : str, default "bce"
            Name of the loss function. Choose between "bce" (binary
            cross entropy) and "nll" (negative log-likelihood). Note
            that nll loss is only allowed for semi-weakly models.
        distribute_strategy : tf.distribute.Strategy
            Strategy used for distributed (multi-GPU) training.
        verbosity : str, default "INFO"
            Verbosity level ("DEBUG", "INFO", "WARNING" or "ERROR").
        """
        super().__init__(feature_level=feature_level,
                         decay_modes=decay_modes,
                         variables=variables,
                         noise_dimension=noise_dimension,
                         use_validation=use_validation,
                         distribute_strategy=distribute_strategy,
                         outdir=outdir,
                         verbosity=verbosity,
                         **kwargs)
        self._loss = loss
        self._sigmoid_activation = sigmoid_activation

    @property
    def loss(self) -> str:
        return self._loss

    @loss.setter
    def loss(self, value:str):
        value = value.lower()
        if value not in ['bce', 'nll']:
            raise ValueError(f'Invalid name for loss function: {value}. '
                             f'Please choose between "bce" and "nll".')
        self._loss = value

    def _distributed_wrapper(self, fn, **kwargs):
        if self.distribute_strategy:
            with self.distribute_strategy.scope():
                result = fn(**kwargs)
        else:
            result = fn(**kwargs)
        return result

    def get_supervised_model_inputs(self, feature_metadata: Dict, downcast: bool = True):
        """
        Get the inputs for a supervised model.

        Parameters
        ----------------------------------------------------
        feature_metadata : dict
            Metadata for the features.
        downcast : bool, default = True
            Whether to downcast float64 to float32.

        Returns
        ----------------------------------------------------
        inputs : dict
            A dictionary of input layers.
        """
        from tensorflow.keras.layers import Input
        
        label_map = {
            'part_coords': 'points',
            'part_features': 'features',
            'part_masks': 'masks'
        }

        tmp_metadata = feature_metadata.copy()
        if downcast:
            for metadata in tmp_metadata.values():
                if metadata['dtype'] == 'float64':
                    metadata['dtype'] = 'float32'

        if self.variables is not None:
            nvar = len(self.variables)
            tmp_metadata['jet_features']['shape'][-1] = nvar

        if self.noise_dimension_per_jet:
            tmp_metadata['jet_features']['shape'][-1] += self.noise_dimension_per_jet

        inputs = {}
        for feature, metadata in tmp_metadata.items():
            key = label_map.get(feature, feature)
            inputs[key] = Input(**metadata, name=feature)
        return inputs

    def get_loss_fn(self, model_type: Optional[Union[str, ModelType]] = None):
        if self.loss == 'bce':
            if model_type and ModelType.parse(model_type) in [SEMI_WEAKLY, IDEAL_WEAKLY]:
                from aliad.interface.tensorflow.losses import ScaledBinaryCrossentropy
                return ScaledBinaryCrossentropy(offset=-np.log(2), scale=1000)
            return 'binary_crossentropy'
        elif self.loss == 'nll':
            if model_type and ModelType.parse(model_type) not in [SEMI_WEAKLY]:
                raise RuntimeError(f'NLL loss is only allowed for semi-weakly models, '
                                   f'not {model_type}.')
            from aliad.interface.tensorflow.losses import ScaledNLLLoss
            return ScaledNLLLoss(offset=0., scale=1.)
        raise ValueError(f'Invalid name for loss function: {self.loss}')
        
    def get_train_config(
        self,
        checkpoint_dir: str,
        model_type: Optional[Union[str, ModelType]] = None,
        weight_clipping: bool = True,
        epochs: Optional[int] = None
    ):
        """
        Get the configuration for training.

        Parameters
        ----------------------------------------------------
        checkpoint_dir : str
            Directory for checkpoints.
        model_type : (optional) str or ModelType 
            The type of model.
        weight_clipping : bool
            Whether to apply weight clipping.

        Returns
        ----------------------------------------------------
        config: dictionary
            The training configuration.
        """
        if model_type and ModelType.parse(model_type) == PRIOR_RATIO:
            config = {
                'loss': 'MSE',
                'epochs': epochs or 3000,
                'optimizer': 'Adam',
                'optimizer_config': {'learning_rate': 0.01},
                'checkpoint_dir': checkpoint_dir,
                'callbacks': {
                    'early_stopping': {
                        'monitor': 'val_loss',
                        'patience': 200,
                        'restore_best_weights': True,
                        'always_restore_best_weights': True
                    }
                }
            }
            return config
            
        if self.feature_level == HIGH_LEVEL:
            epochs = epochs or 100
            patience = 10
        elif self.feature_level == LOW_LEVEL:
            epochs = epochs or 20
            patience = 5
        else:
            raise RuntimeError(f'Unknown feature level: {self.feature_level.key}')

        monitor_metric = 'val_loss' if self.use_validation else 'loss'
        
        metrics = ['accuracy']
        config = {
            'loss': self.get_loss_fn(model_type),
            'metrics': metrics,
            'epochs': epochs,
            'optimizer': 'Adam',
            'optimizer_config': {'learning_rate': 0.001},
            'checkpoint_dir': checkpoint_dir,
            'callbacks': {
                'lr_scheduler': {
                    'initial_lr': 0.001,
                    'lr_decay_factor': 0.5,
                    'patience': 5,
                    'min_lr': 1e-6
                },
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': patience,
                    'restore_best_weights': True,
                    'always_restore_best_weights': True
                },
                'model_checkpoint': {
                    'save_weights_only': True,
                    'save_best_only': True,
                    'save_freq': 'epoch'
                },
                'metrics_logger': {'save_freq': 'epoch'}
            }
        }

        if model_type and ModelType.parse(model_type) == SEMI_WEAKLY:
            
            config['callbacks']['weights_logger'] = {
                'save_freq': 'epoch',
                'display_weight': True
            }                            
            
            lr = 0.05
            if weight_clipping:
                config['optimizer_config'].update({
                    'learning_rate': lr,
                    'clipvalue': 0.0001,
                    'clipnorm': 0.0001
                })
            config['callbacks']['early_stopping']['patience'] = 20
            config['callbacks']['lr_scheduler'] = {
                'initial_lr': lr,
                'lr_decay_factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6,
                'verbose': True
            }
        return config

    def _print_config_summary(self, config):
        self.stdout.info('Train configuration:')
        loss = config['loss'].name if not isinstance(config['loss'], str) else config['loss']
                  
        summary = (
            f"               Optimizer: {config['optimizer']}\n"
            f"       Optimizer Options: {config['optimizer_config']}\n"
            f"           Loss Function: {loss}\n"
            f" Early Stopping Patience: {config['callbacks']['early_stopping']['patience']}\n"
            f"    LR Scheduler Options: {config['callbacks']['lr_scheduler']}"
        )
        self.stdout.info(summary, bare=True)

    def _get_prior_ratio_model(self, feature_metadata: Dict):
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Dense, Normalization
        
        all_inputs = self.get_supervised_model_inputs(feature_metadata)
        param_feature = self._get_param_feature()
        x = all_inputs[param_feature]
        inputs = [x]
        normalizer = Normalization()
        x = normalizer(x)
        for nodes, activation in PRIOR_RATIO_NET_LAYERS:
            x = Dense(nodes, activation)(x)
        model = Model(inputs=inputs, outputs=x, name='PriorRatioNet')
        return model, normalizer
        
    def _get_high_level_model(self, feature_metadata: Dict, parametric: bool = True):
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Dense
        import tensorflow as tf

        all_inputs = self.get_supervised_model_inputs(feature_metadata)

        x1 = all_inputs['jet_features']
        if parametric:
            param_feature = self._get_param_feature()
            x2 = all_inputs[param_feature]
            inputs = [x1, x2]
                                                                   
            x = tf.concat([x1, tf.expand_dims(x2, axis=-1)], -1)
                                
            x = tf.reshape(x, (-1, tf.reduce_prod(tf.shape(x)[1:])))
        else:
            inputs = [x1]
            x = tf.reshape(x1, (-1, tf.reduce_prod(tf.shape(x1)[1:])))

                                 
        for nodes, activation in MLP_LAYERS:
            x = Dense(nodes, activation)(x)
        
        model = Model(inputs=inputs, outputs=x, name='HighLevel')
        
        return model
    
    def _get_low_level_model(self, feature_metadata: Dict, parametric: bool = True):
        from aliad.interface.tensorflow.models import MultiParticleNet
        all_inputs = self.get_supervised_model_inputs(feature_metadata)
        keys = ['points', 'features', 'masks', 'jet_features']
        if parametric:
            param_feature = self._get_param_feature()
            all_inputs['param_features'] = all_inputs[param_feature]
            keys.append('param_features')
        inputs = {key: all_inputs[key] for key in keys}
        model_builder = MultiParticleNet()
        model = model_builder.get_model(**inputs)
        return model
            
    def get_supervised_model(self, feature_metadata: Dict, parametric: bool):
        """
        Get the supervised model.

        Parameters
        ----------------------------------------------------
        feature_metadata : dict
            Metadata for the features.
        parametric : bool
            Whether to include parametric features.

        Returns
        ----------------------------------------------------
        model : keras.Model
            The supervised model.
        """
        if self.feature_level == HIGH_LEVEL:
            model_fn = self._get_high_level_model
        elif self.feature_level == LOW_LEVEL:
            model_fn = self._get_low_level_model
                  
        kwargs = {'feature_metadata': feature_metadata, 'parametric': parametric}
                                    
         
        return self._distributed_wrapper(model_fn, **kwargs)

    @staticmethod
    def get_single_parameter_model(activation: str = 'linear',
                                   kernel_initializer = None,
                                   kernel_constraint = None,
                                   kernel_regularizer = None,
                                   trainable: bool = True,
                                   name: Optional[str] = 'dense'):
        """
        Get a single parameter model.

        Parameters
        ----------------------------------------------------
        activation : str
            Activation function.
        exponential : bool
            Whether to apply exponential activation. Default is False.
        kernel_initializer : keras.Initializer
            Initializer for the kernel.
        kernel_constraint : keras.Constraint
            Constraint for the kernel.
        kernel_regularizer : keras.Regularizer
            Regularizer for the kernel.
        trainable : bool
            Whether the parameter is trainable. Default is True.
        name : str
            Name of the layer.

        Returns
        ----------------------------------------------------
        model : Keras model
            The single-parameter model.
        """
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import Dense
        import tensorflow as tf
        if get_module_version('keras') > (3, 0, 0):
            from keras.ops import exp
        else:
            exp = tf.exp

        inputs = Input(shape=(1,))
        outputs = Dense(1, use_bias=False, activation=activation,
                        kernel_initializer=kernel_initializer,
                        kernel_constraint=kernel_constraint,
                        kernel_regularizer=kernel_regularizer,
                        name=name)(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        if not trainable:
            model.trainable = False
        return model

    @semistaticmethod
    def get_semi_weakly_weights(self, m1: float, m2: float,
                                mu: Optional[float] = None,
                                alpha: Optional[float] = None,
                                use_sigmoid: bool = False):
        """
        Get the weight parameters for constructing the semi-weakly model.

        Parameters
        ----------------------------------------------------
        m1 : float
            Initial value of the first mass parameter (mX).
        m2 : float
            Initial value of the second mass parameter (mY).
        mu : (optional) mu
            Initial value of the signal fraction parameter.
        alpha : (optional) mu
            Initial value of the branching fraction parameter.

        Returns
        ----------------------------------------------------
        weights: dictionary
            Dictionary of weights.
        """
        import tensorflow as tf

        weights = {
            'm1': self.get_single_parameter_model(activation=get_parameter_transform('m1'),
                                                  kernel_initializer=tf.constant_initializer(float(m1)),
                                                  kernel_regularizer=get_parameter_regularizer('m1'),
                                                  name='m1'),
            'm2': self.get_single_parameter_model(activation=get_parameter_transform('m2'),
                                                  kernel_initializer=tf.constant_initializer(float(m2)),
                                                  kernel_regularizer=get_parameter_regularizer('m2'),
                                                  name='m2')
        }
        if mu is not None:
            weights['mu'] = self.get_single_parameter_model(activation=get_parameter_transform('mu'),
                                                            kernel_initializer=tf.constant_initializer(float(mu)),
                                                            kernel_regularizer=get_parameter_regularizer('mu'),
                                                            name='mu')
            
        if alpha is not None:
            weights['alpha'] = self.get_single_parameter_model(activation=get_parameter_transform('alpha'),
                                                               kernel_initializer=tf.constant_initializer(float(alpha)),
                                                               kernel_regularizer=get_parameter_regularizer('alpha'),
                                                               name='alpha')
        return weights

    @staticmethod
    def _get_one_signal_semi_weakly_layer(fs_out, mu,
                                          kappa: float = 1.,
                                          epsilon: float = 1e-6,
                                          bug_fix: bool = True):
        LLR = kappa * fs_out / (1. - fs_out + epsilon)
        LLR_xs = 1. + mu * (LLR - 1.)
        if bug_fix:
            ws_out = LLR_xs / (LLR_xs + 1 - mu)
        else:
            ws_out = LLR_xs / (LLR_xs + 1)
        return ws_out

    @staticmethod
    def _get_two_signal_semi_weakly_layer(fs_2_out, fs_3_out, mu, alpha,
                                          kappa_2: float = 1.,
                                          kappa_3: float = 1.,
                                          epsilon: float = 1e-5,
                                          bug_fix: bool = True):
        LLR_2 = kappa_2 * fs_2_out / (1. - fs_2_out + epsilon)
        LLR_3 = kappa_3 * fs_3_out / (1. - fs_3_out + epsilon)
        LLR_xs = 1. + mu * (alpha * LLR_3 + (1 - alpha) * LLR_2 - 1.)
        if bug_fix:
            ws_out = LLR_xs / (LLR_xs + 1 - mu)
        else:
            ws_out = LLR_xs / (LLR_xs + 1)
        return ws_out

    @staticmethod
    def _get_one_signal_likelihood_layer(fs_out, mu,
                                         kappa: float = 1.,
                                         epsilon: float = 1e-6):
        LLR = kappa * fs_out / (1. - fs_out + epsilon)
        LLR_xs = 1. + mu * (LLR - 1.)
        return LLR_xs  

    @staticmethod
    def _get_two_signal_likelihood_layer(fs_2_out, fs_3_out, mu, alpha,
                                         kappa_2: float = 1.,
                                         kappa_3: float = 1.,
                                         epsilon: float = 1e-6):
        LLR_2 = kappa_2 * fs_2_out / (1. - fs_2_out + epsilon)
        LLR_3 = kappa_3 * fs_3_out / (1. - fs_3_out + epsilon)
        LLR_xs = 1. + mu * (alpha * LLR_3 + (1 - alpha) * LLR_2 - 1.)
        return LLR_xs

    def _get_semi_weakly_model(self, feature_metadata: Dict, fs_model_path: str,
                               m1: float = 0., m2: float = 0.,
                               mu: float = INIT_MU, alpha: float = INIT_ALPHA,
                               kappa: Union[str, float] = INIT_KAPPA,
                               fs_model_path_2: Optional[str] = None,
                               epsilon: float = 1e-6, 
                               bug_fix: bool = True,
                               use_sigmoid: bool = False) -> "keras.Model":
        import tensorflow as tf
        if get_module_version('keras') > (3, 0, 0):
            from keras.ops import ones_like
        else:
            ones_like = tf.ones_like

        inputs = self.get_supervised_model_inputs(feature_metadata)
        weights = self.get_semi_weakly_weights(m1=m1, m2=m2, mu=mu, alpha=alpha, use_sigmoid=use_sigmoid)
        m1_out = weights['m1'](ones_like(inputs['jet_features'])[:, 0, 0])
        m2_out = weights['m2'](ones_like(inputs['jet_features'])[:, 0, 0])
        mu_out = weights['mu'](ones_like(inputs['jet_features'])[:, 0, 0])
        alpha_out = weights['alpha'](ones_like(inputs['jet_features'])[:, 0, 0])
        mass_params = tf.keras.layers.concatenate([m1_out, m2_out])

        train_features = self._get_train_features(SEMI_WEAKLY)
        train_inputs = [inputs[feature] for feature in train_features]
        fs_inputs = [inputs[feature] for feature in train_features]
        fs_inputs.append(mass_params)

        multi_signal = len(self.decay_modes) > 1
        if multi_signal and fs_model_path_2 is None:
            raise ValueError('fs_model_path_2 cannot be None when multiple signals are considered')

        def get_kappa_out(val: Union[str, float], supervised_model_path: str, name: Optional[str] = None):
            if isinstance(val, Number):
                return float(val)
            assert isinstance(val, str)
            val = val.lower()
            if val in ['inferred', 'sampled']:
                basename = self.path_manager.get_file("model_prior_ratio",
                                                      basename_only=True,
                                                      sampling_method=val)
                dirname = os.path.dirname(supervised_model_path)
                model_path = os.path.join(dirname, basename)
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f'prior ratio model path does not exist: {model_path}')
                prior_model = self.load_model(model_path)
                if name is not None:
                    prior_model._name = name
                self.freeze_all_layers(prior_model)
                return prior_model(mass_params)
            return float(val)
            

        if not multi_signal:
            fs_model = self.load_model(fs_model_path)
            fs_model._name = f"{fs_model.name}_1"
            self.freeze_all_layers(fs_model)
            kappa_out = get_kappa_out(kappa, fs_model_path)
            fs_out = fs_model(fs_inputs)
            if self.loss != 'nll':
                ws_out = self._get_one_signal_semi_weakly_layer(fs_out, mu=mu_out, kappa=kappa_out,
                                                                epsilon=epsilon, bug_fix=bug_fix)
            else:
                ws_out = self._get_one_signal_likelihood_layer(fs_out, mu=mu_out, kappa=kappa_out,
                                                               epsilon=epsilon)
            LLR = kappa_out * fs_out / (1 - fs_out + 1e-6)
            self.llr_model = tf.keras.Model(inputs=train_inputs, outputs=LLR, name='LLR')
            self.fs_model = tf.keras.Model(inputs=train_inputs, outputs=fs_out, name='Supervised')
        else:
            if isinstance(kappa, str):
                tokens = split_str(kappa, sep=',', remove_empty=True)
                if len(tokens) == 1:
                    kappa_2, kappa_3 = tokens[0], tokens[0]
                elif len(tokens) == 2:
                    kappa_2, kappa_3 = tokens
                else:
                    raise ValueError(f'failed to interpret kappa value: {kappa}')
            else:
                kappa_2, kappa_3 = kappa, kappa

            fs_2_model = self.load_model(fs_model_path)
            fs_2_model._name = f"{fs_2_model.name}_2prong"
            self.freeze_all_layers(fs_2_model)
            fs_3_model = self.load_model(fs_model_path_2)
            fs_3_model._name = f"{fs_3_model.name}_3prong"
            self.freeze_all_layers(fs_3_model)
            kappa_2_out = get_kappa_out(kappa_2, fs_model_path, "PriorRatioNet_2")
            kappa_3_out = get_kappa_out(kappa_3, fs_model_path_2, "PriorRatioNet_3")
            fs_2_out = fs_2_model(fs_inputs)
            fs_3_out = fs_3_model(fs_inputs)
            if self.loss != 'nll':
                ws_out = self._get_two_signal_semi_weakly_layer(fs_2_out, fs_3_out, mu=mu_out,
                                                                alpha=alpha_out, epsilon=epsilon,
                                                                kappa_2=kappa_2_out, kappa_3=kappa_3_out,
                                                                bug_fix=bug_fix)
            else:
                ws_out = self._get_two_signal_likelihood_layer(fs_2_out, fs_3_out, mu=mu_out,
                                                               alpha=alpha_out, epsilon=epsilon,
                                                               kappa_2=kappa_2_out, kappa_3=kappa_3_out)
            LLR_2 = kappa_out * fs_2_out / (1 - fs_2_out + 1e-6)
            LLR_3 = kappa_out * fs_3_out / (1 - fs_3_out + 1e-6)
            self.llr_2_model = tf.keras.Model(inputs=train_inputs, outputs=LLR_2, name='TwoProngLLR')
            self.llr_3_model = tf.keras.Model(inputs=train_inputs, outputs=LLR_3, name='ThreeProngLLR')
            self.fs_2_model = tf.keras.Model(inputs=train_inputs, outputs=fs_2_out, name='TwoProngSupervised')
            self.fs_3_model = tf.keras.Model(inputs=train_inputs, outputs=fs_3_out, name='ThreeProngSupervised')

        ws_model = tf.keras.Model(inputs=train_inputs, outputs=ws_out, name='SemiWeakly')
        
        return ws_model

    def get_prior_ratio_model(self, feature_metadata: Dict) -> "keras.Model":
        return self._get_prior_ratio_model(feature_metadata)

    def get_semi_weakly_model(self, feature_metadata: Dict, fs_model_path: str,
                              m1: float = 0., m2: float = 0.,
                              mu: float = INIT_MU, alpha: float = INIT_ALPHA,
                              kappa: Union[float, str] = INIT_KAPPA,
                              fs_model_path_2: Optional[str] = None,
                              epsilon: float = 1e-5,
                              bug_fix: bool = True,
                              use_sigmoid: bool = False) -> "keras.Model":
        """
        Get the semi-weakly model.

        Parameters
        ----------------------------------------------------
        feature_metadata: dict
            Metadata for the features.
        fs_model_path: str
            Path to the fully supervised model.
        m1 : float, default 0.
            Initial value of the first mass parameter (mX). This value
            is expected to be overriden later in the training.
        m2 : float, default 0.
            Initial value of the second mass parameter (mY). This value
            is expected to be overriden later in the training.
        mu : float, optional
            Initial value of the signal fraction parameter.
        alpha : float, optional
            Initial value of the branching fraction parameter.
        kappa : float or str, default 1.0
        fs_model_path_2 : str, optional
            Path to the (3-prong) fully supervised model when
            both 2-prong and 3-prong signals are used.
        epsilon : float, default 1e-5.
            Small constant added to the model to avoid division by zero.

        Returns
        ----------------------------------------------------
        model : Keras model
            The semi-weakly model.
        """
        kwargs = {
            'feature_metadata': feature_metadata,
            'fs_model_path': fs_model_path,
            'm1': m1,
            'm2': m2,
            'mu': mu,
            'alpha': alpha,
            'kappa': kappa,
            'fs_model_path_2': fs_model_path_2,
            'epsilon': epsilon,
            'bug_fix': bug_fix,
            'use_sigmoid': use_sigmoid
        }
        model_fn = self._get_semi_weakly_model
        return self._distributed_wrapper(model_fn, **kwargs)

    @staticmethod
    def set_semi_weakly_model_weights(ws_model, m1: Optional[float] = None,
                                      m2: Optional[float] = None,
                                      mu: Optional[float] = None,
                                      alpha: Optional[float] = None) -> None:
        """
        Set the weights for the semi-weakly model. Only parameters with non-None values wil be updated.

        Parameters
        ----------------------------------------------------
        ws_model: Keras model
            The semi-weakly model.
        m1 : (optional) float
            Value of the first mass parameter (mX).
        m2 : (optional) float
            Value of the second mass parameter (mY).
        mu : (optional) float
            Value of the signal fraction parameter.
        alpha : (optional) float
            Value of the branching fraction parameter.
        """
        weight_dict = {
            'm1/kernel:0': m1,
            'm2/kernel:0': m2,
            'mu/kernel:0': mu,
            'alpha/kernel:0': alpha
        }
        for weight in ws_model.trainable_weights:
            name = weight.name
            if name not in weight_dict:
                raise RuntimeError(f'Unknown model weight: {name}. Please make sure model weights are initialized with the proper names')
                                                                                 
                                                  
            value = weight_dict[name]
            if value is not None:
                assign_weight(weight, value)

    @staticmethod
    def get_semi_weakly_model_weights(ws_model) -> Dict:
        """
        Get the weights for the semi-weakly model.

        Parameters
        ----------------------------------------------------
        ws_model: Keras model
            The semi-weakly model.

        Returns
        ----------------------------------------------------
        weights: dictionary
            A dictionary of weights.
        """
        weights = {}
        for weight in ws_model.trainable_weights:
            name = weight.name.split('/')[0]
            value = weight.value().numpy().flatten()[0]
            weights[name] = value
        return weights

    @staticmethod
    def set_model_weights(model, values: Dict) -> None:
        """
        Set the weights for a model.

        Parameters
        ----------------------------------------------------
        model : Keras model
            The model for setting the weights.
        values : dict
            A dictionary mapping the weight name to the weight values.
        """
        weights = model.trainable_weights
                               
        if isinstance(values, dict):
            for weight in weights:
                name = weight.name.split('/')[0]
                if name in values:
                    value = values[name]
                    assign_weight(weight, value)
        else:
            for i, value in enumerate(values):
                assign_weight(weights[i], value)

    @staticmethod
    def compile_model(model, config: Dict) -> None:
        """
        Compile the model with the given configuration.

        Parameters
        ----------------------------------------------------
        model : Keras model
            The model to compile.
        config : dictionary
            A dictionary containing the configuration for compiling the model.
        """
        import tensorflow as tf
        optimizer = getattr(tf.keras.optimizers, config['optimizer'])(**config['optimizer_config'])
        metrics = config['metrics'] if 'metrics' in config else None
        model.compile(loss=config['loss'], optimizer=optimizer, metrics=metrics)

    @staticmethod
    def load_model(model_path: str) -> "keras.Model":
        """
        Load a tensorflow keras model from the specified path.

        Parameters
        ----------------------------------------------------
        model_path : str
            Path to the model.

        Returns
        ----------------------------------------------------
        Model : Keras model
            Loaded model.
        """
        from aliad.interface.keras import load_model
        model = load_model(model_path)
        return model

    @staticmethod
    def freeze_all_layers(model) -> None:
        """
        Freeze all layers of the model.

        Parameters
        ----------------------------------------------------
        model : Keras model
            The model whose layers to freeze.
        """
        for layer in model.layers:
            layer.trainable = False

    @staticmethod
    def freeze_model(model) -> None:
        """
        Freeze the entire model.

        Parameters
        ----------------------------------------------------
        model : Keras model
            The model to freeze.
        """
        model.trainable = False    

    def get_callbacks(
        self,
        model_type: Union[str, ModelType],
        config: Dict,
        targets: Optional[List[str]] = None
    ) -> Dict:
        """
        Get the callbacks for training.

        Parameters
        ----------------------------------------------------
        model_type : str or ModelType
            The type of model.
        config : dict
            Configuration dictionary.

        Returns
        ----------------------------------------------------
        callbacks : Dict
            Dictionary of callbacks.
        """
        from aliad.interface.tensorflow.callbacks import LearningRateScheduler, MetricsLogger, WeightsLogger, EarlyStopping
                                                                                       
        from tensorflow.keras.callbacks import ModelCheckpoint
        
        checkpoint_dir = config['checkpoint_dir']

        callbacks = {}

        targets = list(targets) if targets is not None else list(config['callbacks'])
        if 'early_stopping' in targets:
            callbacks['early_stopping'] = EarlyStopping(**config['callbacks']['early_stopping'])
            
        if 'model_checkpoint' in targets:
            model_ckpt_filepath = os.path.join(checkpoint_dir,
                                               self.path_manager.get_basename('model_checkpoint'))
            callbacks['model_checkpoint'] = ModelCheckpoint(model_ckpt_filepath, **config['callbacks']['model_checkpoint'])
            
        if 'metrics_logger' in targets:
            callbacks['metrics_logger'] = MetricsLogger(checkpoint_dir, **config['callbacks']['metrics_logger'])

        if 'lr_scheduler' in targets:
            lr_scheduler = LearningRateScheduler(**config['callbacks']['lr_scheduler'])
            callbacks['lr_scheduler'] = lr_scheduler

        model_type = ModelType.parse(model_type)
        if (model_type == SEMI_WEAKLY) and ('weights_logger' in targets):
            weights_logger = WeightsLogger(checkpoint_dir, **config['callbacks']['weights_logger'])
            callbacks['weights_logger'] = weights_logger

        return callbacks

    @semistaticmethod
    def restore_model(self, early_stopping, model, checkpoint_dir: str) -> None:
        """
        Restore the model from a checkpoint.

        Parameters
        ----------------------------------------------------
        early_stopping : EarlyStopping
            Early stopping callback.
        model : Keras model
            The model to restore.
        checkpoint_dir : str 
            Directory for checkpoints.
        """
        metrics_ckpt_filepath = os.path.join(checkpoint_dir,
                                             self.path_manager.get_basename("metrics_checkpoint"))
        model_ckpt_filepath = os.path.join(checkpoint_dir,
                                           self.path_manager.get_basename("model_checkpoint"))
        early_stopping.restore(model, metrics_ckpt_filepath=metrics_ckpt_filepath,
                               model_ckpt_filepath=model_ckpt_filepath)