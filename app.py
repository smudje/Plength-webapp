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
import random
import string
from processor import Processor

app = Flask(__name__)
app.secret_key = 'Plength'
app.debug = True
UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])
filename = ''

@app.route('/')
def hello_world():
  return render_template('base.html')

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploadImage', methods=['GET', 'POST'])
def upload_file():
  if request.data is None:
    return ''
  else:
    # app.logger.debug(request.data)
    # f = open("imgData.txt", "wb")
    # f.write(request.data)
    # f.close()
    #app.logger.debug(request.method)
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    with open(UPLOAD_FOLDER + filename + ".png", "wb") as fh:
      fh.write(base64.decodebytes(request.data))

   
@app.route('/test', methods=['GET', 'POST'])
def test_analyze():
  prcsr = Processor()
  prcsr.setfile("uploads\XVMPSA.png")
  prcsr.analyze()
  return prcsr.test()