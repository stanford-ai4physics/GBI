from .settings import (
    SIGMOID_ACTIVATION,
    MASS_SCALE,
    MASS_RANGE,
    SEMI_WEAKLY_PARAMETERS
)

def get_parameter_transform(parameter: str, backend: str = 'tensorflow'):
    from aliad.components import activations

    if parameter in ['m1', 'm2']:
        return activations.Scale(1 / MASS_SCALE, backend=backend)
    elif parameter in ['mu', 'alpha']:
        if SIGMOID_ACTIVATION:
            return activations.Sigmoid(backend=backend)
        if parameter == 'mu':
            return activations.Exponential(backend=backend)
        return activations.Linear(backend=backend)
    raise ValueError(f'unknown parameter for semi-weakly model: {parameter}')

def get_parameter_inverse_transform(parameter: str, backend: str = 'tensorflow'):
    return get_parameter_transform(parameter, backend=backend).inverse

def get_parameter_transforms(backend: str = 'tensorflow'):
    transforms = {}
    for parameter in SEMI_WEAKLY_PARAMETERS:
        transforms[parameter] = get_parameter_transform(parameter, backend=backend)
    return transforms

def get_parameter_inverse_transforms(backend: str = 'tensorflow'):
    inverse_transforms = {}
    for parameter in SEMI_WEAKLY_PARAMETERS:
        inverse_transforms[parameter] = get_parameter_inverse_transform(parameter, backend=backend)
    return inverse_transforms

def get_parameter_regularizer(parameter: str):
    from aliad.interface.keras.regularizers import MinMaxRegularizer
    
    if parameter in ['m1', 'm2']:
        mass_range = (85 * MASS_SCALE, MASS_RANGE[1] * MASS_SCALE, 1)
        return MinMaxRegularizer(*mass_range)
    elif parameter == 'mu':
        if SIGMOID_ACTIVATION:
            return MinMaxRegularizer(-10, -2, 10)
        return MinMaxRegularizer(-10.0, 0.0, 10)
    elif parameter == 'alpha':
        if SIGMOID_ACTIVATION:
            return None
        return MinMaxRegularizer(0.0, 1.0, 10.0)
    raise ValueError(f'unknown parameter for semi-weakly model: {parameter}')