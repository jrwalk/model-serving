import pytest

import datetime
import pandas

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn_pandas import DataFrameMapper


# model pipeline fixture
@pytest.fixture
def dummy_pipeline():
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
def dummy_data():
    df = pandas.DataFrame({"X": [1], "Y": [1]})
    y = [1]
    return (df, y)


@pytest.fixture
def dummy_pipeline_trained(dummy_data, dummy_pipeline):
    df, y = dummy_data
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
