from flask import Blueprint
data_blueprint = Blueprint(
    'data', __name__, template_folder='templates')

from . import routes