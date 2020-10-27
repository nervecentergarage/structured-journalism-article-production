from flask import render_template
from flask import request
from project import mongo
from flask import jsonify
import re

from . import content_blueprint


@content_blueprint.route('/publishSummary/', methods=['POST'])
def publishSummary():
    return ("Hello from publishSummary")


@content_blueprint.route('/generateArticle/', methods=['POST'])
def generateArticle():
    return ("Hello from generateArticle")
