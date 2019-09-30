from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from randomMiniBatchKS import random_mini_batches
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import uuid

# load the user configs
with open('conf/conf.json') as f:    
  config = json.load(f)

# config variables
test_size     = config["test_size"]
seed      = config["seed"]
features_path   = config["features_path"]
labels_path   = config["labels_path"]
results     = config["results"]
classifier_path = config["classifier_path"]
train_path    = config["train_path"]
num_classes   = config["num_classes"]
classifier_path = config["classifier_path"]

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)
labels = np.reshape(labels, (labels.shape[0],1))

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

num_epoch = 700

with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('vgg16_KS-{}.meta'.format(num_epoch))
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))
	print("Model restored from file")
	
    sess.run(tf.global_variables_initializer())
         all_vars = tf.get_collection('vars')
             for v in all_vars:
             	v_ = sess.run(v)
             	print(v_)

