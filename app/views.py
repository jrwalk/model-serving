import cloudpickle as pickle

from .model_utils import (
    build_prediction,
    parse_model,
    build_data
)

with open("/model-serving/binary/pipeline.pkl", 'rb') as rf:
    _model = pickle.load(rf)


def get_params():
    """build parsed model metadata view for model read into view memory.

    RETURNS:
        :param (dict):
            model metadata dictionary.
    """
    return parse_model(_model)


def predict(input):
    """Builds prediction from input data for model read into view memory.

    ARGS:
        :param (dict) input:
            input data JSON.

    RETURNS:
        :param (tuple[list, list]):
            output predictions + probabilities.
    """
    df = build_data(input)
    return build_prediction(_model, df)
