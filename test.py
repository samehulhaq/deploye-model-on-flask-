# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:37:31 2021

@author: TAC
"""

import sys
import os
import glob
import re
import numpy as np

# Keras
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu,True)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)

model.summary()
