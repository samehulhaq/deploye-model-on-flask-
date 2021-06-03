from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/fruits_checkpoints.h5'

# Load your trained model
model = load_model(MODEL_PATH)
        # Necessary
# print('Model loaded. Start serving...')
encode={'Apricot': 0,
 'Avocado': 1,
 'Banana': 2,
 'Chestnut': 3,
 'Clementine': 4,
 'Granadilla': 5,
 'Kiwi': 6,
 'Limes': 7,
 'Mango': 8,
 'Maracuja': 9,
 'Peach': 10,
 'Pear': 11,
 'Pomegranate': 12,
 'Raspberry': 13,
 'Pineapple': 14,
 'Strawberry': 15,
 'Walnut': 16}

decode={v:k for k,v in encode.items()}
# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
	img=image.load_img(img_path,target_size=(35, 35))
	img=image.img_to_array(img)
	img=np.expand_dims(img, axis=0)
	pred=model.predict(img)
	pred=np.argmax(pred)
	classname=decode[pred]
	return classname


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax

        return preds
    return None
 

if __name__ == '__main__':
    app.run(debug=True)

