# import CORS as CORS
from flask import Flask
from flask_cors import CORS
import create_model

app = Flask(__name__)
# app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


# cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:4200"}})
# CORS(app, resources={r"/http://localhost": {"origins": "http://localhost"}})

@app.route("/GetFilePath/<path>", methods=['GET', 'POST'])
# @cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def GetFilePath(path):
    print(path)
    create_model.train_model(path)
    return path
