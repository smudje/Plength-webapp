# Initial Routing App to test packages
from flask import (
  Flask,
  render_template,
  flash,
  request,
  jsonify,
  send_file
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
    fullpath = UPLOAD_FOLDER + filename + ".png" 
    with open(fullpath, "wb") as fh:
      fh.write(base64.decodebytes(request.data))

    result = analyze(fullpath, filename)
    if (result == True):
      sendfile = UPLOAD_FOLDER + "F_" + filename + ".png"
      #return send_file(sendfile, mimetype='image/png')
      with open(sendfile, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string
    else:
      return "Error"


    
def analyze(path, filename):
  prcsr = Processor()
  prcsr.setfile(path, filename)
  result = prcsr.analyze()
  return result


   
@app.route('/test', methods=['GET', 'POST'])
def test_analyze():
  prcsr = Processor()
  prcsr.setfile("uploads\XVMPSA.png")
  prcsr.analyze()
  return prcsr.test()