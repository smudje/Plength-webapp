# Initial Routing App to test packages
from flask import (
  Flask,
  render_template,
  flash,
  request,
  jsonify
)
from werkzeug.utils import secure_filename
import base64
import os 

app = Flask(__name__)
app.secret_key = 'Plength'
app.debug = True
UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

@app.route('/')
def hello_world():
  return render_template('base.html')

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploadImage', methods=['GET', 'POST'])
def upload_file():
  # app.logger.debug(request.data)
  f = open("imgData.txt", "wb")
  f.write(request.data)
  f.close()
  #app.logger.debug(request.method)
  with open("imageToSave.png", "wb") as fh:
    fh.write(base64.decodebytes(request.data))