# model-serving

A simple Dockerized flask API for serving ML models.  This has mainly been an
exercise in
[test-driven development](https://en.wikipedia.org/wiki/Test-driven_development)
practices, and a place to tinker with `docker-compose` for container
development.

## the model

For simplicity, the API uses scikit-learn [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
objects combined with the [`sklearn-pandas`](https://github.com/scikit-learn-contrib/sklearn-pandas)
package's `DataFrameMapper` to systematize the model interface.  The pipeline
must begin with (at least one) `DataFrameMapper` object - this encodes any
preprocessing transforms needed, and allows feature selection by name from the
input JSON.  Conversely, the last step of the pipeline must be an
`sklearn.BaseEstimator` instance for the final prediction (any steps in between
need only support the `sklearn` transformer API).

## Docker usage

The service is built on top of the `python3.7-slim` image using `docker-compose`, which provides three services:

1. `box` - base build service and general-purpose debugging interface to
the container.
2. `test` - Runs the `pytest` suite of unit tests for the service.
3. `app` - starts the Flask debug webserver.

Simply running `docker-compose up` from the root directory will trigger all three services.

To load a model into new container, simply ensure that the model is packed via
`cloudpickle` into a file called `pipeline.pkl` in the `./binary` directory - this is mounted as a volume to the container and is immediately accessible.

## the API

### get API usage

This endpoint retrieves general usage for the API.

#### HTTP request

`GET <url:port>/usage`

#### parameters

None.

### get model metadata

This endpoint retrieves a JSON payload of model metadata - expected features,
step names, and model type.

#### HTTP request

`GET <url:port>/model`

#### parameters

None.

### download model

This endpoint downloads a `cloudpickle`-serialized copy of the pipeline.

#### HTTP request

`GET <url:port>/model/download`

#### parameters

None.

### generate predictions

This endpoint runs model inference on the supplied data.

#### HTTP request

`POST <url:port>/model/predict`

#### parameters

```
{
    "data": {
        "<column 1>": [...],
        "<column 2>": [...],
        ...
    }
}
```

## requirements

Docker >2.0, `docker-compose`, and

```atomicwrites==1.2.1
attrs==18.2.0
Click==7.0
cloudpickle==0.6.1
Flask==1.0.2
itsdangerous==0.24
Jinja2==2.10.1
Markdown==3.1
MarkupSafe==1.0
more-itertools==4.3.0
numpy==1.15.2
pandas==0.23.4
pluggy==0.8.0
py==1.7.0
pytest==3.9.1
python-dateutil==2.7.5
pytz==2018.7
scikit-learn==0.20.0
scipy==1.1.0
six==1.11.0
sklearn-pandas==1.7.0
Werkzeug==0.14.1
```

## to-dos & next steps

1. additional "production-grade" service (gunicorn or similar)
2. transition to using [FastAPI](https://fastapi.tiangolo.com/) for
aynchronous serving (includes `uvicorn` webserver)
3. more unit testing!
4. better support for model upload/control
