import pytest

import datetime
import pandas

from app.app import app

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn_pandas import DataFrameMapper


@pytest.fixture
def client():
    context = app.app_context()
    context.push()
    yield app.test_client()
    context.pop()


# model pipeline fixture
@pytest.fixture
def dummy_pipeline():
    """base pipeline using `sklearn.DummyClassifier`
    """
    dummy_mapper = DataFrameMapper(features=[
        ("X", None),
        (["X", "Y"], None)
    ], input_df=True, df_out=True)
    clf = DummyClassifier(strategy='stratified', random_state=42)
    pipe = Pipeline(memory=None,
                    steps=[
                        ("Mapper", dummy_mapper),
                        ("Classifier", clf)
                    ])
    return pipe


@pytest.fixture
def bad_pipeline_no_mapper():
    """pipeline missing its `DataFrameMapper`.  Will throw an error.
    """
    clf = DummyClassifier(strategy="stratified", random_state=42)
    return Pipeline(memory=None,
                    steps=[
                        ("Classifier", clf)
                    ])


@pytest.fixture
def bad_pipeline_no_model():
    """pipeline missing its `sklearn` model.  Will throw an error.
    """
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
    """single sample data
    """
    df = pandas.DataFrame({"X": [1], "Y": [1]})
    y = [1]
    y_prob = [0, 1]
    return (df, y, y_prob)


@pytest.fixture
def dummy_data_multi():
    """multi sample data
    """
    df = pandas.DataFrame({"X": [1, 0], "Y": [1, 0]})
    y = [1, 0]
    y_prob = [[0, 1], [1, 0]]
    return (df, y, y_prob)


@pytest.fixture
def bad_data():
    """single sample of malformed data.  Will throw an error.
    """
    df = pandas.DataFrame({"X": [None], "Y": [None]})
    return (df, None, None)


@pytest.fixture
def missing_data():
    """Missing column from the data.  Will throw an error.
    """
    df = pandas.DataFrame({"X": [1]})
    return (df, None, None)


@pytest.fixture
def dummy_pipeline_trained(dummy_data_multi, dummy_pipeline):
    """dummy pipeline with trained flag set.
    """
    df, y, yp = dummy_data_multi
    dummy_pipeline.fit(df, y)
    return dummy_pipeline


@pytest.fixture
def dummy_pipeline_logistic(dummy_data_multi):
    """compensate for DummyClassifier not failing on bad data,
    so we'll use a real model here
    """
    dummy_mapper = DataFrameMapper(features=[
        ("X", None),
        ("Y", None)
    ], input_df=True, df_out=True)
    clf = LogisticRegression()
    pipe = Pipeline(memory=None,
                    steps=[
                        ("Mapper", dummy_mapper),
                        ("Classifier", clf)
                    ])

    df, y, yp = dummy_data_multi
    pipe.fit(df, y)
    return pipe


# timing fixtures
_dummy_time = datetime.datetime(2018, 1, 1)


@pytest.fixture
def dummy_time():
    return _dummy_time


@pytest.fixture
def patch_datetime(monkeypatch):
    """monkeypatch for check to upload time attribute.
    """
    class mydatetime:
        @classmethod
        def utcnow(cls):
            return _dummy_time

    monkeypatch.setattr(datetime, 'datetime', mydatetime)
