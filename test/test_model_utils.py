import pytest

import cloudpickle as pickle

from app.model_utils import (
    parse_model,
    build_data,
    build_prediction
)


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
    assert model.get("model") == "LogisticRegression"


def test_parse_model_no_pipeline():
    with pytest.raises(TypeError) as e:
        parse_model(None)
    assert "Pipeline" in str(e.value)


def test_parse_model_missing_mapper(bad_pipeline_no_mapper):
    with pytest.raises(ValueError) as e:
        parse_model(bad_pipeline_no_mapper)
    assert "DataFrameMapper" in str(e.value)


def test_parse_model_missing_model(bad_pipeline_no_model):
    with pytest.raises(ValueError) as e:
        parse_model(bad_pipeline_no_model)
    assert "Estimator" in str(e.value)


# model predict stage tests
def test_model_predict_untrained(dummy_pipeline, dummy_data_single):
    df, y = dummy_data_single

    with pytest.raises(ValueError) as e:
        build_prediction(dummy_pipeline, df)
    assert "untrained" in str(e.value)


def test_model_predict_missing_data(dummy_pipeline_trained, missing_data):
    df, y = missing_data

    with pytest.raises(KeyError) as e:
        build_prediction(dummy_pipeline_trained, df)
    assert "not in index" in str(e.value)


def test_model_predict_bad_data(dummy_pipeline_trained, bad_data):
    df, y = bad_data

    with pytest.raises(ValueError) as e:
        build_prediction(dummy_pipeline_trained, df)
    assert "misformatted" in str(e.value)


def test_model_predict_single(dummy_pipeline_trained, dummy_data_single):
    df, y = dummy_data_single
    assert build_prediction(dummy_pipeline_trained, df) == y[0]


def test_model_predict_multi(dummy_pipeline_trained, dummy_data_multi):
    df, y = dummy_data_multi
    assert build_prediction(dummy_pipeline_trained, df) == y
