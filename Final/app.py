import os
from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
import glob
import functions as f
import deployed

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2048 * 2048
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
app.config['UPLOAD_PATH'] = 'static/uploads'
app.config['IMAGE_PATH'] = 'static/images'

# code to download and store all images; no longer necessary after run for first time
# f.download_images(f.create_dict('static/image_links.csv')[0], "static/images/")

# code to load all necessary info for each plant disease
link_dict=f.create_link_dict("static/info_links.csv")
plant_disease_dict, diseases = f.create_dict('static/disease_info.csv')

# results=['Potato Late Blight', 'Tomato Leaf Mold', 'Tomato Late Blight', 'Peach Healthy']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display')
def display():
    files = os.listdir(app.config['UPLOAD_PATH'])
    results=[]
    for file in files:
        results.append(diseases[deployed.predict_new(os.path.join(app.config['UPLOAD_PATH'], file))])
    return render_template('display.html', files=files, pd_dict=plant_disease_dict, results=results, num=len(files), link_dict=link_dict)

@app.route('/', methods=['POST'])
def upload_files():
    files = glob.glob(os.path.join(app.config['UPLOAD_PATH'], '*'))
    for f in files:
        os.remove(f)
    uploaded_files = request.files.getlist('image_file')
    for uploaded_file in uploaded_files:
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                abort(400)
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
    return redirect('/display')

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/images/<filename>')
def image(filename):
    return send_from_directory(app.config['IMAGE_PATH'], filename)

if __name__ == '__main__':
    app.run(debug=True)