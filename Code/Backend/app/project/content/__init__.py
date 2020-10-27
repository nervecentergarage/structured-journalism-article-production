from flask import Blueprint
content_blueprint = Blueprint(
    'content', __name__, template_folder='templates')

from . import routes