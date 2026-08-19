import os
from flask import Flask
from flask_pymongo import PyMongo
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()
mongo = PyMongo()


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'f5f58d08c51419aca1208bc5a68467fd')

    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        raise RuntimeError(
            "MONGO_URI is not set.\n"
            "Copy .env.example to .env and fill in your MongoDB Atlas connection string."
        )
    app.config['MONGO_URI'] = mongo_uri
    mongo.init_app(app)

    app.mongo = mongo

    from .auth import auth as auth_blueprint
    from .views import views as views_blueprint
    from .recognition import recognition as recognition_blueprint
    from .teacher import teacher_bp as teacher_blueprint

    app.register_blueprint(auth_blueprint)
    app.register_blueprint(views_blueprint)
    app.register_blueprint(recognition_blueprint, url_prefix='/recognition')
    app.register_blueprint(teacher_blueprint)

    return app
