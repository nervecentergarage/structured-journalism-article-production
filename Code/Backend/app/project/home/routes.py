from . import home_blueprint
from flask import render_template


@home_blueprint.route('/', methods=['GET'])
def home():
    return "Hello World"
