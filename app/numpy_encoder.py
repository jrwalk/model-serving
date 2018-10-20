import json
import numpy as np


_int_types = (
    np.int_, np.intc, np.intp, np.int8, 
    np.int16, np.int32, np.int64, np.uint8,
    np.uint16, np.uint32, np.uint64
    )

_float_types = (np.float_, np.float16, np.float32, np.float64)


class NumpyEncoder(json.JSONEncoder):
    """override JSON encoder to handle numpy type casting.
    """
    def default(self, obj):
        if isinstance(obj, _int_types):
            return int(obj)
        elif isinstance(obj, _float_types):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(obj)