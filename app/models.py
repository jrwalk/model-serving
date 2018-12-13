from sklearn_pandas import DataFrameMapper
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import datetime as dt
import collections


def flatten(l):
    for el in l:
        if (isinstance(el, collections.Iterable) and
                not isinstance(el, (str, bytes))):
            yield from flatten(el)
        else:
            yield el


def parse_model(model):
    """m
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


def predict(model, df):
    pass
