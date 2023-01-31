import os
import cv2
import shutil
import numpy as np
from PIL import Image
from skimage import color
from skimage.color import rgb2gray

import tensorflow as tf
from keras.utils import save_img
from flask import Flask, request, render_template
from Data import Data

model_dir = os.path.join('static', 'Final_Model.h5')
data_dir = os.path.join('static', 'input')
output_dir = os.path.join('static', 'output')

DIMENSION = (256, 256)
RESAMPLE = 3

app = Flask(__name__, static_folder="static")
app.config['GRAYSCALE_DIR'] = os.path.join('static', 'input')
app.config['COLORIZED_DIR'] = os.path.join('static', 'output')

@app.errorhandler(413)
def too_large(e):
		return "File is too large", 413

@app.route('/')
def index():
	url = request.host_url
	if url[4] != 's':
		url = url[:4] + 's' + url[4:]
	return render_template('index.html', hostUrl=url)

@app.route('/colorize', methods=['POST'])
def colorize():
	# 1). Empty upload directory
	emptyDirectory(app.config['GRAYSCALE_DIR'])

	# 3). Get image and save in backend
	if ('grayscale_image' in request.files):
		try:
			file = request.files['grayscale_image']
			file.save(os.path.join(app.config['GRAYSCALE_DIR'], file.filename))

			dataSet = Data(data_dir, DIMENSION, RESAMPLE)
			dataSet.loadImage()
			dataSet.processTestData()

			# Load Model
			model = tf.keras.models.load_model(model_dir)

			model_output = model.predict(dataSet.l_org_test, verbose=0)
			model_output *= 100

			predicted_images = []

			for i in range(len(model_output)):
				predicted_lab = np.concatenate((dataSet.l_org_test[i], model_output[i]),axis=2)
				predicted_rgb = np.asarray(color.lab2rgb(predicted_lab)) * 255
				predicted_rgb = np.uint8(predicted_rgb)
				predicted_rgb = Image.fromarray(predicted_rgb).resize(dataSet.org_size[i], resample=RESAMPLE)

				predicted_images.append(predicted_rgb)

			colorized_image_path = ''
			for i in range(len(predicted_images)):
				filename = dataSet.file_names[i]
				extension = filename[filename.find('.'):]
				colorized_image_path = os.path.join(output_dir, 'colorized_image' + extension)
				save_img(colorized_image_path, predicted_images[i])
				grayscale_image_path = os.path.join(output_dir, 'grayscale_image' + extension)
				cv2.imwrite(grayscale_image_path, dataSet.grayscaleImages[i])

			colorized_image_string = ''
			grayscale_image_string = ''
			if os.path.isfile(colorized_image_path):
				import base64
				with open(colorized_image_path, "rb") as img_file:
					colorized_image_string = base64.b64encode(img_file.read()).decode("utf-8")
			else:
				errMsg = f"Cannot find outputted colorized image in backend!'"
				print('Failed: ' + errMsg)
				return {'data': [], 'status': 'failed', 'errMsg': errMsg}

			if os.path.isfile(grayscale_image_path):
				import base64
				with open(grayscale_image_path, "rb") as img_file:
					grayscale_image_string = base64.b64encode(img_file.read()).decode("utf-8")
			else:
				errMsg = f"Cannot find outputted grayscale image in backend!'"
				print('Failed: ' + errMsg)
				return {'data': [], 'status': 'failed', 'errMsg': errMsg}
			
			return_data = {
				'colorized_image': colorized_image_string,
				'grayscale_image': grayscale_image_string
			}

			return {'data': return_data, 'status': 'ok', 'errMsg': ''}
		except Exception as e:
			errMsg = 'Save image file in backend failed! ' + e
			print('Failed: ' + errMsg)
			return {'data': [], 'status': 'failed', 'errMsg': errMsg}
	else:
		warnMsg = 'No image data to predict.'
		print('Warning: ' + warnMsg)
		return {'data': [], 'status': 'warning', 'errMsg': warnMsg}
	


def emptyDirectory(directory):
	for filename in os.listdir(directory):
		file_path = os.path.join(directory, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			errMsg = f'Empty directory "{directory}" failed! ' + str(e)
			print('Failed: ' + errMsg)
			return {'data': [], 'status': 'failed', 'errMsg': errMsg}

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)