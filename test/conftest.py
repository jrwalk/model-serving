import pytest

import datetime
import pandas

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper


# model pipeline fixture
@pytest.fixture
def dummy_pipeline():
    dummy_mapper = DataFrameMapper(features=[
        ("X", None),
        (["X", "Y"], None)
    ], input_df=True, df_out=True)
    # clf = DummyClassifier(strategy='stratified', random_state=42)
    clf = LogisticRegression()
    pipe = Pipeline(memory=None,
                    steps=[
                        ("Mapper", dummy_mapper),
                        ("Classifier", clf)
                    ])
    return pipe


@pytest.fixture
def bad_pipeline_no_mapper():
    clf = DummyClassifier(strategy="stratified", random_state=42)
    return Pipeline(memory=None,
                    steps=[
                        ("Classifier", clf)
                    ])


@pytest.fixture
def bad_pipeline_no_model():
    dummy_mapper = DataFrameMapper(features=[
        ("X", None),
        (["X", "Y"], None)
    ], input_df=True, df_out=True)
    return Pipeline(memory=None,
                    steps=[
                        ("Mapper", dummy_mapper)
                    ])


@pytest.fixture
def dummy_data_single():
    df = pandas.DataFrame({"X": [1], "Y": [1]})
    y = [1]
    return (df, y)


@pytest.fixture
def dummy_data_multi():
    df = pandas.DataFrame({"X": [1, 0], "Y": [1, 0]})
    y = [1, 0]
    return (df, y)


@pytest.fixture
def bad_data():
    df = pandas.DataFrame({"X": ['a'], "Y": [None]})
    y = [1]
    return (df, y)


@pytest.fixture
def missing_data():
    df = pandas.DataFrame({"X": [1]})
    y = [1]
    return (df, y)


@pytest.fixture
def dummy_pipeline_trained(dummy_data_multi, dummy_pipeline):
    df, y = dummy_data_multi
    dummy_pipeline.fit(df, y)
    return dummy_pipeline


# timing fixtures
_dummy_time = datetime.datetime(2018, 1, 1)


@pytest.fixture
def dummy_time():
    return _dummy_time


@pytest.fixture
def patch_datetime(monkeypatch):
    class mydatetime:
        @classmethod
        def utcnow(cls):
            return _dummy_time

    monkeypatch.setattr(datetime, 'datetime', mydatetime)
