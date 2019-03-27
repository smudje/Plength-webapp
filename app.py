# Initial Routing App to test packages
from flask import (
  Flask,
  render_template,
  flash,
  request
)
from werkzeug.utils import secure_filename

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
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return 'Py Error'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return 'File saved'