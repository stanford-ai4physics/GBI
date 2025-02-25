from typing import Union, Optional, Dict, List, TypeVar

from quickstats.core.typing import Numeric, ArrayLike
from aliad.interface.keras.utils import is_keras_model

T = TypeVar('T')

KappaInputType = Union[Numeric, T]

# "prob", "likelihood"

class PAWS:

    @property
    def priors(self) -> Union[T, Dict[str, T]]:
        return self._priors

    @property
    def kappas(self) -> Optional[Union[KappaInputType, Dict[str, KappaInputType]]]=None,
        return self._kappas
	
	def __init__(
        self,
        priors: Union[T, Dict[str, T]],
        parameters: ...,
        kappas: Optional[Union[KappaInputType, Dict[str, KappaInputType]]]=None,
        output_mode: str = "likelihood"
    ):
        self._priors = priors

    def validate_prior(
        self,
        prior: T
    ) -> T:
        if not is_keras_model(prior):
            raise ValueError(f'Prior must be a Keras model')
        return prior
        
    def validate_priors(
        self,
        priors: Union[T, Dict[str, T]]
    ):
        if isinstance(priors, dict):
            valid_priors = {}
            for key in priors:
                valid_priors[key] = self.validate_prior(priors[key])
            return valid_priors
        return self.validate_prior(priors)
            
    def validate_kappas(
        self,
        kappas: Optional[Union[KappaInputType, Dict[str, KappaInputType]]]=None
    ):
        pass
        
				 
	def fix_parameters(self, ...):
		pass
		
	def float_parameters(self, ...):
		pass
	
	def set_parameters(self, ...):
		pass
		
	def parameters(self, ...):
		pass
		
	def fit(self, ...):
		pass
		
	def summary(self, ...):
		pass

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


    def _create_model(self):
        pass

    def _get_semi_weakly_model(self, feature_metadata: Dict, fs_model_path: str,
                               m1: float = 0., m2: float = 0.,
                               mu: float = INIT_MU, alpha: float = INIT_ALPHA,
                               kappa: Union[str, float] = INIT_KAPPA,
                               fs_model_path_2: Optional[str] = None,
                               epsilon: float = 1e-6, 
                               bug_fix: bool = True,
                               use_sigmoid: bool = False) -> "keras.Model":
        import tensorflow as tf
        from aliad.interface.keras.ops import ones_like

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
            self.fs_2_model = tf.keras.Model(inputs=train_inputs, outputs=fs_2_out, name='TwoProngSupervised')
            self.fs_3_model = tf.keras.Model(inputs=train_inputs, outputs=fs_3_out, name='ThreeProngSupervised')

        ws_model = tf.keras.Model(inputs=train_inputs, outputs=ws_out, name='SemiWeakly')
        
        return ws_model

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