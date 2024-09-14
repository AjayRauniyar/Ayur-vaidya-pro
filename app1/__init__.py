from flask import Blueprint

app1 = Blueprint('app1', __name__, template_folder='templates')

from app1 import app1_routes 
