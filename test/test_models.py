import pytest

import cloudpickle as pickle
import pandas

from app import parse_model, predict


# model upload tests
def test_pickleable_model(dummy_pipeline):
    assert pickle.loads(pickle.dumps(dummy_pipeline))


def test_parse_model(dummy_pipeline):
    assert parse_model(dummy_pipeline) is not None


def test_parse_model_args(dummy_pipeline):
    model = parse_model(dummy_pipeline)
    assert set(model.get('args')) == {"X", "Y"}


def test_parse_model_steps(dummy_pipeline):
    model = parse_model(dummy_pipeline)
    assert model.get('steps') == ["Mapper", "Classifier"]


def test_parse_model_timestamp(patch_datetime, dummy_time, dummy_pipeline):
    model = parse_model(dummy_pipeline)
    assert model.get('uploaded') == dummy_time


def test_parse_model_model(dummy_pipeline):
    model = parse_model(dummy_pipeline)
    assert model.get("model") == "DummyClassifier"


def test_parse_model_missing_mapper(bad_pipeline_no_mapper):
    with pytest.raises(ValueError) as e:
        parse_model(bad_pipeline_no_mapper)
    assert "DataFrameMapper" in str(e.value)


def test_parse_model_missing_model(bad_pipeline_no_model):
    with pytest.raises(ValueError) as e:
        parse_model(bad_pipeline_no_model)
    assert "Estimator" in str(e.value)


# model predict stage tests
def test_model_predict_untrained(dummy_pipeline, dummy_data):
    df, y = dummy_data

    with pytest.raises(ValueError) as e:
        dummy_pipeline.predict(df)
    assert "untrained" in str(e.value)


def test_model_predict_wrong_data(dummy_pipeline_trained, dummy_data):
    df, y = dummy_data
    df.drop(['X'], axis=1, inplace=True)

    with pytest.raises(ValueError) as e:
        dummy_pipeline_trained.predict(df)
    assert "missing" in str(e.value)


def test_model_predict_single(dummy_pipeline_trained, dummy_data):
    df, y = dummy_data
    assert predict(dummy_pipeline_trained, df) == y[0]


def test_model_predict_multi(dummy_pipeline_trained, dummy_data):
    df, y = dummy_data
    df = pandas.concat([df, df], ignore_index=True)
    y = y * 2
    assert predict(dummy_pipeline_trained, df) == y
