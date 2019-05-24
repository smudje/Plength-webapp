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
import threading

app = Flask(__name__)
app.secret_key = 'Plength'
app.debug = True
UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

filename = ''
prcsr = Processor()

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
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    fullpath = UPLOAD_FOLDER + filename + ".png"
    
    image = request.form['image'].encode()
    calPos = request.form['calPos']
    dist = int(request.form['distance'])
    minDist = int(request.form['minimumDist'])
    plantType = request.form['plantType']

    print(plantType)

    with open(fullpath, "wb") as fh:
      fh.write(base64.decodebytes(image))

    if plantType == "c":
      plantType = "coleoptile"
    else:
      plantType = "seedling"

    if calPos == "l":
      side = "left"
    else:
      side = "right"
 
    result, data = analyze(fullpath, filename, plantType, side, minDist, dist)
    if (result == True):
      sendfile = UPLOAD_FOLDER + "F_" + filename + ".png"
      #return send_file(sendfile, mimetype='image/png')
      with open(sendfile, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read())
      
      return image_base64
    else:
      return "Error"


    
def analyze(path, filename, plantType, side, minDist, dist):
  #prcsr = Processor()
  prcsr.setfile(path, filename)
  result = prcsr.analyze(plantType, side, dist  , minDist)
  return result

@app.route('/getData', methods=['GET', 'POST'])
def getData():
  return jsonify(data=prcsr.getData())
   
@app.route('/pollProgress', methods=['GET', 'POST'])
def pollProgress():
  return jsonify(progress=prcsr.getProgress())