from flask import Flask
from main_blueprint import main
from app1.app1_routes import app1  # Ensure proper import path
from app2.app2 import app2 
from ayur3.app3_routes import app3
# from app3.app3_routes import app3 # Ensure proper import path

def create_app():
    app = Flask(__name__)
    app.register_blueprint(main, url_prefix='/')
    app.register_blueprint(app3, url_prefix='/home')
    app.register_blueprint(app1, url_prefix='/medsign')
    app.register_blueprint(app2, url_prefix='/home2')
    # app.register_blueprint(app3, url_prefix='/medbot')
    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=3011)
