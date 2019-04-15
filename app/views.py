import cloudpickle as pickle

from .model_utils import (
    build_prediction,
    parse_model,
    build_data
)

with open("/model-serving/binary/pipeline.pkl", 'rb') as rf:
    _model = pickle.load(rf)


def get_params():
    return parse_model(_model)


def predict(input):
    """
    """
    df = build_data(input)
    return build_prediction(_model, df)
