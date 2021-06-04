from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
#memroy handling check https://github.com/samehulhaq/deploye-model-on-flask-/issues/2
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

#read encoded dictionary and decode it
with open('dict.txt','r') as d:
    encode=d.read()    
import json
json_acceptable_string = encode.replace("'", "\"")
encode = json.loads(json_acceptable_string)

decode={v:k for k,v in encode.items()}


print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
	img=image.load_img(img_path,target_size=(224, 224))
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

        ##Make prediction
        preds = model_predict(file_path, model)
        return preds
 

if __name__ == '__main__':
    app.run(debug=True)

