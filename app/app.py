from flask import Flask

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