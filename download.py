from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
# import matplotlib.pyplot as plt

# # Scikit-learn includes many helpful utilities
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle

import re
# import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

# from keras.layers.normalization import layer_normalization


path = '/home/ivsh/datasets'
annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=path,
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
                                          extract = True)
annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2017.json'

name_of_zip = 'train2017.zip'
if not os.path.exists(path + '/' + name_of_zip):
  image_zip = tf.keras.utils.get_file(name_of_zip,
                                      cache_subdir=path,
                                      origin = 'http://images.cocodataset.org/zips/train2017.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip)+'/train2017/'
else:
  PATH = path+'/train2017/'

# path = '/home/ivsh/datasets'
# annotation_zip = tf.keras.utils.get_file('captions.zip',
#                                           cache_subdir=path,
#                                           origin = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
#                                           extract = True)
annotation_file_val = os.path.dirname(annotation_zip)+'/annotations/captions_train2017.json'

name_of_zip = 'train2017.zip'
if not os.path.exists(path + '/' + name_of_zip):
  image_zip = tf.keras.utils.get_file(name_of_zip,
                                      cache_subdir=path,
                                      origin = 'http://images.cocodataset.org/zips/train2017.zip',
                                      extract = True)
  PATH = os.path.dirname(image_zip)+'/train2017/'
else:
  PATH = path+'/train2017/'

