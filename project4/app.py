import joblib
import numpy as np
import torch
import torchvision
from flask import Flask
from flask_restful import Api, Resource, reqparse

from model import get_prediction

APP = Flask(__name__)
API = Api(APP)


class Predict(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument("image_path")

        args = parser.parse_args()  # creates dict

        out = {"Prediction": get_prediction(args["image_path"])}

        return out, 200


API.add_resource(Predict, "/predict")

if __name__ == "__main__":
    APP.run(debug=True, port=5000)
