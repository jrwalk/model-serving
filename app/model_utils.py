import pandas
from sklearn_pandas import DataFrameMapper
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import datetime as dt
import collections


def flatten(l):
    """flatten a nested list of strings.

    ARGS:
        :param (list) l:
            nested list of strings.

    RETURNS:
        :param (list):
            flattened list.
    """
    for el in l:
        if (isinstance(el, collections.Iterable) and
                not isinstance(el, (str, bytes))):
            yield from flatten(el)
        else:
            yield el


def build_data(input):
    """build `pandas.DataFrame` from input JSON.

    ARGS:
        :param (dict) input:
            input JSON.

    RETURNS:
        :param (`pandas.DataFrame`):
            converted `DataFrame` instance.
    """
    return pandas.DataFrame.from_dict(input, orient='columns')


def parse_model(model):
    """extract metadata from `sklearn.Pipeline` instance.

    ARGS:
        :param (`sklearn.Pipeline`) model:
            input model instance.

    RETURNS:
        :param (dict):
            model metadata.

    RAISES:
        `TypeError`: for malformed pipelines
        `ValueError`: for missing steps in the pipeline
    """
    try:
        assert isinstance(model, Pipeline)
        steps = [s[0] for s in model.steps]
    except AssertionError as e:
        raise TypeError("uploaded model must be an",
                        " `sklearn.pipeline.Pipeline`")
    except AttributeError as e:
        raise TypeError("uploaded model is not a ",
                        "valid `sklearn.pipeline.Pipeline`")

    timestamp = dt.datetime.utcnow()

    try:
        mapper = model.steps[0][1]
        assert isinstance(mapper, DataFrameMapper)
        features = [f[0] for f in mapper.features]
        features = list(set(flatten(features)))
    except AssertionError:
        raise ValueError("input stage of pipeline must be a `DataFrameMapper`")
    except Exception:
        raise ValueError("malformed `DataFrameMapper` in pipeline")

    try:
        predictor = model.steps[-1][1]
        assert (isinstance(predictor, BaseEstimator) and
                not isinstance(predictor, DataFrameMapper))
        modtype = str(predictor).split('(')[0]
    except AssertionError:
        raise ValueError("end stage of pipeline must be sklearn Estimator")
    except Exception:
        raise ValueError("malformed sklearn Estimator in pipeline")

    return {
        "steps": steps,
        "uploaded": timestamp,
        "model": modtype,
        "args": features
    }


def build_prediction(model, df):
    """build prediction for pipeline.  Returns single sample + probas
    for model prediction, or list of predictions for multi-sample.

    ARGS:
        :param (`sklearn.Pipeline`) model:
            model instance.
        :param (`pandas.DataFrame`) df:
            input data.

    RETURNS:
        :param (tuple):
            output predictions + probabilities

    RAISES:
        `KeyError`: for missing input data
        `ValueError`: for misformatted data or untrained models.
    """
    try:
        y = model.predict(df)
        y_prob = model.predict_proba(df)
        if len(df) == 1:
            return (y[0], y_prob[0].tolist())
        else:
            return (y.tolist(), y_prob.tolist())
    except KeyError as e:
        raise KeyError("missing data: {}".format(e))
    except TypeError:   # catches untrained models before the classifier stage
        raise ValueError("model is untrained")
    except ValueError:
        raise ValueError("misformatted data passed to model")
