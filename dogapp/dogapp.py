import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash
from werkzeug.utils import secure_filename

from .detector import *

DIR_PATH = os.getcwd()
UPLOAD_FOLDER = os.path.join(DIR_PATH,'dogapp','static','images')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/', methods=['GET','POST'])
def upload_file():
	if request.method == 'POST':
		#check if post request has imagefile
		if 'imagefile' not in request.files:
			#flash('No file uploaded')
			return redirect(request.url)
		imagefile = request.files['imagefile']
		if imagefile.filename == '':
			#flash('No selected file')
			return redirect(request.url)
		if imagefile and allowed_file(imagefile.filename):
			filename = secure_filename(imagefile.filename)
			imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			#return render_template('form.html')
			return redirect(url_for('.results', filename=filename))

	return render_template('form.html')


@app.route('/results')
def results():
	filename = os.path.join('images',request.args['filename'])
	breed = dog_breed_detector(os.path.join('dogapp','static',filename))
	breed = breed.replace('_',' ').title()
	filename = url_for('static', filename=filename)
	return render_template('results.html', filename=filename, breed_results=breed)

