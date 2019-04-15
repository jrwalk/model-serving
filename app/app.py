from flask import Flask, request


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


@app.route("/models")
def get_models():
    pass


@app.route("/models/<model_id>")
def get_model(model_id):
    pass


@app.route("/models/<model_id>/predict")
def predict_model(model_id):
    pass
