import json
import decimal
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, (complex, np.complexfloating)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

ctx = decimal.Context()
ctx.prec = 20

# taken from https://stackoverflow.com/questions/38847690
def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def str_encode_value(val:float, n_digit=None, formatted=True):
    if n_digit is not None:
        val_str = '{{:.{}f}}'.format(n_digit).format(val)
    else:
        val_str = float_to_str(val)
    # edge case of negative zero
    if val_str == '-0.0':
        val_str = '0p0'
    
    if formatted:
        val_str = val_str.replace('.', 'p').replace('-', 'n')
    return val_str


def find_zero_crossings(x, y):
    """
    Returns a list of x-values where y crosses zero, 
    using linear interpolation between data points.
    """
    crossings = []
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i+1]
        if y0 == 0:
            # Exactly zero at i
            crossings.append(x[i])
        elif y1 == 0:
            # Exactly zero at i+1
            crossings.append(x[i+1])
        elif y0 * y1 < 0:
            # There's a sign change between i and i+1
            # Do a linear interpolation for a better estimate
            x0, x1 = x[i], x[i+1]
            crossing_x = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossings.append(crossing_x)
    return crossings
