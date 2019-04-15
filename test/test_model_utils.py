import pytest

import cloudpickle as pickle

from app.model_utils import (
    parse_model,
    build_prediction
)


# model upload tests
def test_pickleable_model(dummy_pipeline):
    """ensure model is pickleable.
    """
    assert pickle.loads(pickle.dumps(dummy_pipeline))


def test_parse_model(dummy_pipeline):
    """ensure we can parse names/attributes from model pipeline.
    """
    assert parse_model(dummy_pipeline) is not None


def test_parse_model_args(dummy_pipeline):
    """ensure we can get the flattened args from the pipeline.
    """
    model = parse_model(dummy_pipeline)
    assert set(model.get('args')) == {"X", "Y"}


def test_parse_model_steps(dummy_pipeline):
    """ensure we can get pipeline steps.
    """
    model = parse_model(dummy_pipeline)
    assert model.get('steps') == ["Mapper", "Classifier"]


def test_parse_model_timestamp(patch_datetime,
                               dummy_time,
                               dummy_pipeline):
    """ensure we can get an upload timestamp.
    """
    model = parse_model(dummy_pipeline)
    assert model.get('uploaded') == dummy_time


def test_parse_model_model(dummy_pipeline):
    """ensure we can get a model class from the pipeline.
    """
    model = parse_model(dummy_pipeline)
    assert model.get("model") == "DummyClassifier"


def test_parse_model_no_pipeline():
    """check error value for missing pipeline.
    """
    with pytest.raises(TypeError) as e:
        parse_model(None)
    assert "Pipeline" in str(e.value)


def test_parse_model_missing_mapper(bad_pipeline_no_mapper):
    """check error value for missing `DataFrameMapper`.
    """
    with pytest.raises(ValueError) as e:
        parse_model(bad_pipeline_no_mapper)
    assert "DataFrameMapper" in str(e.value)


def test_parse_model_missing_model(bad_pipeline_no_model):
    """check error value for missing model.
    """
    with pytest.raises(ValueError) as e:
        parse_model(bad_pipeline_no_model)
    assert "Estimator" in str(e.value)


# model predict stage tests
def test_model_predict_untrained(dummy_pipeline, dummy_data_single):
    """check error value for model without `trained` flag set.
    """
    df, y, yp = dummy_data_single

    with pytest.raises(ValueError) as e:
        build_prediction(dummy_pipeline, df)
    assert "untrained" in str(e.value)


def test_model_predict_missing_data(dummy_pipeline_trained, missing_data):
    """check error value for inputs with missing data.
    """
    df, y, yp = missing_data

    with pytest.raises(KeyError) as e:
        build_prediction(dummy_pipeline_trained, df)
    assert "not in index" in str(e.value)


def test_model_predict_bad_data(dummy_pipeline_logistic, bad_data):
    """check error value for malformed data.

    Requires use of real model rather than `DummyClassifier`.
    """
    df, y, yp = bad_data

    with pytest.raises(ValueError) as e:
        build_prediction(dummy_pipeline_logistic, df)
    assert "misformatted" in str(e.value)


def test_model_predict_single(dummy_pipeline_trained, dummy_data_single):
    """check formation for single-sample prediction.
    """
    df, y, yp = dummy_data_single
    pred, pred_prob = build_prediction(dummy_pipeline_trained, df)
    assert (
        (y == pred)
        & (yp == pred_prob)
    )


def test_model_predict_multi(dummy_pipeline_trained, dummy_data_multi):
    """check formation for multi-sample prediction.
    """
    df, y, yp = dummy_data_multi
    pred, pred_prob = build_prediction(dummy_pipeline_trained, df)
    assert (
        (y == pred)
        & (yp == pred_prob)
    )
