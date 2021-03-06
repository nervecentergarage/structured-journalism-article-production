from flask import Flask
from flask_pymongo import PyMongo
from flask_cors import CORS
import nltk
import os
mongo = PyMongo()


def create_app():

    # Flask Config
    app = Flask(__name__)
    initialize_extensions(app)
    register_blueprints(app)
    nltk.download('punkt')
    return app


def initialize_extensions(app):
    CORS(app)
    app.config['MONGO_URI'] = os.environ.get('WEB_MONGO_SNIPPET_DB')
    mongo.init_app(app)


def register_blueprints(app):
    from project.home import home_blueprint
    from project.data import data_blueprint
    from project.content import content_blueprint
    app.register_blueprint(home_blueprint)
    app.register_blueprint(data_blueprint)
    app.register_blueprint(content_blueprint)
