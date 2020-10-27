from flask_cors import CORS
from flask import Flask
from flask_restful import Resource, Api
from project import create_app

app = create_app()

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=False, host='0.0.0.0')
