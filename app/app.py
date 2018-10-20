from flask import Flask
app = Flask(__name__)


@app.route('/')
def home():
    return "this sentence is already halfway over, and still hasn't said anything at all"