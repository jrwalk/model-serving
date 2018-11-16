import pytest
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn_pandas import DataFrameMapper
import cloudpickle as pickle


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


def test_pickleable_model(dummy_pipeline):
    assert pickle.loads(pickle.dumps(dummy_pipeline))


def test_parse_model():
    assert False