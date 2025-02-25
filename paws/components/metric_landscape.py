from typing import Optional, Dict, List
from itertools import repeat
from collections import defaultdict
import numpy as np

from aliad.components.outputs import ModelOutput
from quickstats import AbstractObject, semistaticmethod
from quickstats.parsers import ParamParser
from quickstats.utils.common_utils import execute_multi_tasks

from paws.components.model_loader import ModelLoader
from paws.settings import MASS_SCALE
from paws.utils import get_parameter_transforms

class MetricLandscape(AbstractObject):
    """
    Tool for evaluating metric landscapes of supervised/semi-weakly models.

    Metrics can be loss, auc or accuracy.
    """

    def __init__(self, verbosity: Optional[str] = 'INFO'):
        self.param_transforms = get_parameter_transforms()
        super().__init__(verbosity=verbosity)

    @semistaticmethod
    def _get_y_true(self, dataset):
        data = list(dataset.take(1))
        if len(data[0]) == 2:  # X, Y
            y_index = -1
        elif len(data[0]) == 3:  # X, Y, weight
            y_index = -2
        else:
            raise RuntimeError('Invalid dataset format')
        y_true = np.concatenate([data[y_index] for data in dataset]).flatten()
        return y_true

    def eval_semiweakly(
        self,
        model,
        dataset,
        param_expr: str,
        metrics: Optional[List[str]] = None,
        nbootstrap: Optional[int] = None,
        seed: Optional[int] = None,
        label: Optional[int] = None
    ) -> Dict:
        """
        Get prediction from a semi-weakly model for the given dataset.

        Parameters
        ----------
        model : keras.Model
            The semi-weakly model from which the predictions are made.
        dataset : tf.data.Dataset
            The dataset used for prediction.
        param_expr: str
            An expression specifying the parameter space to scan over.
            The format is "<param_name>=<min_val>_<max_val>_<step>".
            Multi-dimensional space can be specified by joining two
            expressions with a comma. To fix the value of a parameter,
            use the format "<param_name>=<value>". To includ a finite
            set of values, use "<param_name>=(<value_1>,<value_2>,...)".
        metrics: (optional) list of str
            List of metrics to evaluate. If None, the model output as
            well as the truth labels will be saved instead.
        label: (optional) int
            Label corresponding to the signal of interest.

        Returns
        -------
        result : dict
            A dictionary of the predicted outputs and the truth labels.
        """
        init_weights = ModelLoader.get_semi_weakly_model_weights(model)

        param_points = ParamParser.parse_param_str(param_expr)

        y_true = self._get_y_true(dataset)

        parameters = defaultdict(list)
        if not metrics:
            outputs = {
                'y_true': y_true,
                'y_pred': [],
                'params': parameters
            }
            metrics = []
        else:
            outputs = parameters
            
        for param_point in param_points:
            weights = {}
            for key, val in param_point.items():
                if key not in init_weights:
                    raise ValueError(f'Invalid model parameter: "{key}"')
                if val is None:
                    weights[key] = init_weights[key]
                    param_point[key] = self.param_transforms[key](weights[key])
                else:
                    weights[key] = self.param_transforms[key].inverse(val)
                parameters[key].append(param_point[key])
            ModelLoader.set_model_weights(model, weights)
            encoded_str = ParamParser.val_encode_parameters(param_point)
            self.stdout.info(f"Running model prediction with {encoded_str}")

            y_pred = model.predict(dataset).flatten()
            for metric in metrics:
                metric_val = self._evaluate(metric, y_pred, y_true, label=label)
                parameters[metric].append(metric_val)
            if nbootstrap is not None:
                parameters['bootstrap_index'].append(-1)
                parameters['bootstrap_index'].extend(list(range(nbootstrap)))
                for key, val in param_point.items():
                    parameters[key].extend([val] * nbootstrap)
                np.random.seed(seed)
                sample_weights = np.random.poisson(size=(nbootstrap, len(y_pred)))
                for metric in metrics:
                    for sample_weight in sample_weights:
                        metric_val = self._evaluate(metric, y_pred, y_true, weight=sample_weight, label=label)
                        parameters[metric].append(metric_val)
            if not metrics:
                outputs['y_pred'].append(y_pred)
        for key in parameters:
            parameters[key] = np.array(parameters[key])

        return outputs

    def eval_supervised(self, model, dataset,
                        param_expr: str,
                        metrics: Optional[List[str]] = None,
                        label: Optional[int] = None) -> Dict:
        """
        Get prediction from a supervised model for the given dataset.

        Parameters
        ----------
        model : keras.Model
            The supervised model from which the predictions are made.
        dataset : tf.data.Dataset
            The dataset used for prediction.
        param_expr: str
            An expression specifying the parameter space to scan over.
            The format is "<param_name>=<min_val>_<max_val>_<step>".
            Multi-dimensional space can be specified by joining two
            expressions with a comma. To fix the value of a parameter,
            use the format "<param_name>=<value>". To includ a finite
            set of values, use "<param_name>=(<value_1>,<value_2>,...)".
        label: (optional) int
            Label corresponding to the signal of interest.
        
        Returns
        -------
        result : dict
            A dictionary of the predicted outputs and the truth labels.
        """
        import tensorflow as tf
        from tensorflow.keras import Input
        weights = ModelLoader.get_semi_weakly_weights(m1=0., m2=0.)
        features = list(dataset.take(1))[0][0]
        train_inputs = [Input(shape=feature.shape[1:], dtype=feature.dtype) for feature in features]
        dummy_input = tf.ones_like(tf.keras.layers.Flatten()(train_inputs[0]))[:, 0]
        m1_out = self.param_transforms['m1'].inverse(weights['m1'](dummy_input))
        m2_out = self.param_transforms['m1'].inverse(weights['m2'](dummy_input))
        mass_params = tf.keras.layers.concatenate([m1_out, m2_out])
        model_inputs = [inpt for inpt in train_inputs]
        model_inputs.append(mass_params)
        ModelLoader.freeze_all_layers(model)
        model_out = model(model_inputs)
        param_model = tf.keras.Model(inputs=train_inputs, outputs=model_out)

        param_points = ParamParser.parse_param_str(param_expr)

        y_true = self._get_y_true(dataset)
        outputs = {
            'predictions': defaultdict(list)
        }
        if metrics is None:
            outputs['y_true'] = y_true
            metrics = []

        for param_point in param_points:
            for key, val in param_point.items():
                outputs['predictions'][key].append(val)
            encoded_str = ParamParser.val_encode_parameters(param_point)
            ModelLoader.set_model_weights(param_model, param_point)
            self.stdout.info(f"Running model prediction with {encoded_str}")

            y_pred = param_model.predict(dataset).flatten()
            for metric in metrics:
                metric_val = self._evaluate(metric, y_pred, y_true, label=label)
                outputs['predictions'][metric].append(metric_val)
            if not metrics:
                outputs['predictions']['y_pred'].append(y_pred)
        for key in outputs['predictions']:
            outputs['predictions'][key] = np.array(outputs['predictions'][key])

        return outputs

    @staticmethod
    def _evaluate(
        metric: str,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        weight: Optional[np.ndarray] = None,
        label: Optional[int] = None):
        if label is not None:
            mask = y_true == label
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        output = ModelOutput(y_true=y_true, y_pred=y_pred, weight=weight)
        return getattr(output, metric)()

    @semistaticmethod
    def _get_metric_landscape(self, data, metric: str,
                              label: Optional[int] = None,
                              parallel: int = -1):
        y_true = data['y_true']
        predictions = data['predictions']
        # check if awkward array
        if hasattr(predictions, 'fields'):
            parameters = [key for key in predictions.fields if key != 'y_pred']
        else:
            parameters = [key for key in predictions.keys() if key != 'y_pred']
        num_points = len(predictions[parameters[0]])
        landscape = []
        metric_values = execute_multi_tasks(self._evaluate, repeat(metric),
                                            predictions['y_pred'], repeat(y_true),
                                            repeat(label), parallel=parallel)
        for i in range(num_points):
            point = {param: predictions[param][i] for param in parameters}
            point[metric] = metric_values[i]
            landscape.append(point)
        return landscape

    @semistaticmethod
    def get_logloss_landscape(self, data, label: Optional[int] = None, parallel: int = -1):
        return self._get_metric_landscape(data, 'log_loss', label=label, parallel=parallel)

    @semistaticmethod
    def get_max_likelihood_landscape(self, data, label: Optional[int] = None, parallel: int = -1):
        return self._get_metric_landscape(data, 'max_likelihood', label=label, parallel=parallel)

    @semistaticmethod
    def get_mle_landscape(self, data, label: Optional[int] = None, parallel: int = -1):
        return self._get_metric_landscape(data, 'max_likelihood', label=label, parallel=parallel)

    @semistaticmethod
    def get_auc_landscape(self, data, label: Optional[int] = None, parallel: int = -1):
        return self._get_metric_landscape(data, 'auc', label=label, parallel=parallel)

    @semistaticmethod
    def get_accuracy_landscape(self, data, label: Optional[int] = None, parallel: int = -1):
        return self._get_metric_landscape(data, 'accuracy', label=label, parallel=parallel)