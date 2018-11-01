import numpy as np
import json
from app import NumpyEncoder as Encoder


_encoded_json = {
    "an_int": np.int32(0),
    "a_float": np.float64(1),
    "an_array": np.array([1])
}

_json_repr = '{"an_int": 0, "a_float": 1.0, "an_array": [1]}'


def test_encode():
    assert json.dumps(_encoded_json, cls=Encoder) == _json_repr


def test_decode():
    assert json.loads(_json_repr) == _encoded_json