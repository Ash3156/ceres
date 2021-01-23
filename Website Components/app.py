import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2048 * 2048
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
app.config['UPLOAD_PATH'] = 'uploads'

@app.route('/')
def index():
    files = os.listdir(app.config['UPLOAD_PATH'])
    return render_template('index.html', files=files)

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist('image_file')
    for uploaded_file in uploaded_files:
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

if __name__ == '__main__':
    app.run(debug=True)