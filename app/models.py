import pandas
from sklearn_pandas import DataFrameMapper
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
    """
    """
    steps = [s[0] for s in model.steps]
    timestamp = dt.datetime.utcnow()
    modtype = str(model.steps[-1][1]).split('(')[0]
    features = [f[0] for f in model.steps[0][1].features]
    features = list(set(flatten(features)))

    return {
        "steps": steps,
        "uploaded": timestamp,
        "model": modtype,
        "args": features
    }


def predict(model, df):
    pass
