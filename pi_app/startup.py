from flask import Flask
from classes import Classifier

app = Flask(__name__)
model = Classifier()