from flask import Flask
from flask_pymongo import PyMongo
from flask_cors import CORS

mongo = PyMongo()


def create_app():

    # Flask Config
    app = Flask(__name__)
    initialize_extensions(app)
    register_blueprints(app)
    return app


def initialize_extensions(app):
    CORS(app)
    app.config['MONGO_URI'] = " "
    mongo.init_app(app)


def register_blueprints(app):
    from project.home import home_blueprint
    from project.data import data_blueprint
    from project.content import content_blueprint
    app.register_blueprint(home_blueprint)
    app.register_blueprint(data_blueprint)
    app.register_blueprint(content_blueprint)
