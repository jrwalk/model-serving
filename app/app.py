import markdown
from flask import (
    Flask,
    request,
    send_file,
    jsonify,
    render_template,
    Markup
)


from .model_utils import parse_model, build_data
from .views import predict, get_params


# configuration and instantiation
app = Flask(__name__)
app.config['TESTING'] = True
app.config['ENV'] = 'development'


@app.route('/')
def home():
    return ("this sentence is already halfway over,"
            " and still hasn't said anything at all")


@app.route('/usage')
def read_usage():
    with open("README.md", "r") as rf:
        return rf.read()


@app.route('/model', methods=['GET'])
def get_model():
    return jsonify(get_params())


@app.route("/model/download", methods=['GET'])
def download_model():
    return send_file("/model-serving/binary/pipeline.pkl")


@app.route("/model/predict", methods=['POST'])
def predict_model():
    pass
