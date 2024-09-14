from flask import Blueprint

app2 = Blueprint('app2', __name__, template_folder='templates')

from app2  import app2
