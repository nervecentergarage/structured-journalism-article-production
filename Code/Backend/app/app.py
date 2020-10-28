import os
from flask_cors import CORS
from flask import Flask
from flask_restful import Resource, Api
from project import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT'))
    app.run(debug=False, port=port, host='0.0.0.0')
