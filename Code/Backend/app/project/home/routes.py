from . import home_blueprint
from flask import render_template
from tasks import hello


@home_blueprint.route('/', methods=['GET'])
def home():
    hello.delay()
    return 'Hello'
