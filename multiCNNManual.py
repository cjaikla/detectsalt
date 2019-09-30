# filter warnings
from __future__ import print_function

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from randomMiniBatchKS import random_mini_batches4CNN

# keras imports
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.layers import Input
from Image_path import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time
import tensorflow as tf
import re
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import uuid

# load the user configs
with open('conf/conf.json') as f:    
  config = json.load(f)

# config variables
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
features_train_path   = config["features_train_path"]
labels_train_path   = config["labels_train_path"]
features_dev_path   = config["features_dev_path"]
labels_dev_path   = config["labels_dev_path"]
features_test_path   = config["features_test_path"]
labels_test_path   = config["labels_test_path"]
test_size     = config["test_size"]
results     = config["results"]
model_path    = config["model_path"]
seed      = config["seed"]



# start time
print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# import train features and labels
h5f_data  = h5py.File(features_train_path, 'r')
h5f_label = h5py.File(labels_train_path, 'r')

features_string = h5f_data['dataset_train']
labels_string   = h5f_label['dataset_train']

features_train = np.array(features_string)
trainData = np.reshape(features_train,(features_train.shape[0],8,14,14,512))
labels_train   = np.array(labels_string)
trainLabels = np.reshape(labels_train, (labels_train.shape[0],1))

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] train features shape: {}".format(trainData.shape))
print ("[INFO] train labels shape: {}".format(trainLabels.shape))


# import dev features and labels
h5f_data  = h5py.File(features_dev_path, 'r')
h5f_label = h5py.File(labels_dev_path, 'r')

features_string = h5f_data['dataset_dev']
labels_string   = h5f_label['dataset_dev']

features_dev = np.array(features_string)
testData = np.reshape(features_dev,(features_dev.shape[0],8,14,14,512))
labels_dev   = np.array(labels_string)
testLabels = np.reshape(labels_dev, (labels_dev.shape[0],1))

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] dev features shape: {}".format(testData.shape))
print ("[INFO] dev labels shape: {}".format(testLabels.shape))

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not

base_model = VGG16(weights="imagenet")
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
image_size = (224, 224)

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

paramInitWeight = {}

for name, weight in zip(names, weights):
  if name == 'block5_conv1/kernel:0':
      paramInitWeight['B5_C1_K'] = weight
  elif name == 'block5_conv1/bias:0':
      paramInitWeight['B5_C1_B'] = weight
  elif name == 'block5_conv2/kernel:0':
      paramInitWeight['B5_C2_K'] = weight
  elif name == 'block5_conv2/bias:0':
      paramInitWeight['B5_C2_B'] = weight
  elif name == 'block5_conv3/kernel:0':
      paramInitWeight['B5_C3_K'] = weight                
  elif name == 'block5_conv3/bias:0':
      paramInitWeight['B5_C3_B'] = weight
  elif name == 'fc1/kernel:0':
      paramInitWeight['fc1_K'] = weight   
  elif name == 'fc1/bias:0':
      paramInitWeight['fc1_B'] = weight        

with tf.name_scope('inputs'):
  x = tf.placeholder(tf.float32,shape=[None,8,14,14,512])
  y = tf.placeholder(tf.float32,shape=[None,1])
  # x = tf.placeholder(shape=[None, 32768], dtype=tf.float32, name = "x")
  # y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name = "y")

# preparesubdata
# xTrainDataSub1 = tf.reshape(x_input[:,0,:,:,:], [x_input.shape[0],x_input.shape[2],x_input.shape[3],x_input.shape[4]])
xTrainDataSub1 = tf.reshape(x[:,0,:,:,:], [-1,14,14,512])
xTrainDataSub2 = tf.reshape(x[:,1,:,:,:], [-1,14,14,512])
xTrainDataSub3 = tf.reshape(x[:,2,:,:,:], [-1,14,14,512])
xTrainDataSub4 = tf.reshape(x[:,3,:,:,:], [-1,14,14,512])
xTrainDataSub5 = tf.reshape(x[:,4,:,:,:], [-1,14,14,512])
xTrainDataSub6 = tf.reshape(x[:,5,:,:,:], [-1,14,14,512])
xTrainDataSub7 = tf.reshape(x[:,6,:,:,:], [-1,14,14,512])
xTrainDataSub8 = tf.reshape(x[:,7,:,:,:], [-1,14,14,512])

# extract sub1
conv_kernel_1_sub1 = tf.nn.conv2d(xTrainDataSub1, paramInitWeight['B5_C1_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_1_sub1 = tf.nn.bias_add(conv_kernel_1_sub1, paramInitWeight['B5_C1_B'])
layer_1_sub1 = tf.nn.relu(bias_layer_1_sub1)
conv_kernel_2_sub1 = tf.nn.conv2d(layer_1_sub1, paramInitWeight['B5_C2_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_2_sub1 = tf.nn.bias_add(conv_kernel_2_sub1, paramInitWeight['B5_C2_B'])
layer_2_sub1 = tf.nn.relu(bias_layer_2_sub1)
conv_kernel_3_sub1 = tf.nn.conv2d(layer_2_sub1, paramInitWeight['B5_C3_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_3_sub1 = tf.nn.bias_add(conv_kernel_3_sub1, paramInitWeight['B5_C3_B'])
layer_3_sub1 = tf.nn.relu(bias_layer_3_sub1)
last_pool_sub1 = tf.layers.max_pooling2d(inputs=layer_3_sub1, pool_size=[2, 2], strides=2)
last_flattening_sub1 = tf.reshape(last_pool_sub1, [-1, 7*7*512])
extractF_sub1 = tf.nn.relu_layer(last_flattening_sub1, paramInitWeight['fc1_K'], paramInitWeight['fc1_B'])

# extract sub2
conv_kernel_1_sub2 = tf.nn.conv2d(xTrainDataSub2, paramInitWeight['B5_C1_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_1_sub2 = tf.nn.bias_add(conv_kernel_1_sub2, paramInitWeight['B5_C1_B'])
layer_1_sub2 = tf.nn.relu(bias_layer_1_sub2)
conv_kernel_2_sub2 = tf.nn.conv2d(layer_1_sub2, paramInitWeight['B5_C2_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_2_sub2 = tf.nn.bias_add(conv_kernel_2_sub2, paramInitWeight['B5_C2_B'])
layer_2_sub2 = tf.nn.relu(bias_layer_2_sub2)
conv_kernel_3_sub2 = tf.nn.conv2d(layer_2_sub2, paramInitWeight['B5_C3_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_3_sub2 = tf.nn.bias_add(conv_kernel_3_sub2, paramInitWeight['B5_C3_B'])
layer_3_sub2 = tf.nn.relu(bias_layer_3_sub2)
last_pool_sub2 = tf.layers.max_pooling2d(inputs=layer_3_sub2, pool_size=[2, 2], strides=2)
last_flattening_sub2 = tf.reshape(last_pool_sub2, [-1, 7*7*512])
extractF_sub2 = tf.nn.relu_layer(last_flattening_sub2, paramInitWeight['fc1_K'], paramInitWeight['fc1_B']) 

# extract sub3
conv_kernel_1_sub3 = tf.nn.conv2d(xTrainDataSub3, paramInitWeight['B5_C1_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_1_sub3 = tf.nn.bias_add(conv_kernel_1_sub3, paramInitWeight['B5_C1_B'])
layer_1_sub3 = tf.nn.relu(bias_layer_1_sub3)
conv_kernel_2_sub3 = tf.nn.conv2d(layer_1_sub3, paramInitWeight['B5_C2_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_2_sub3 = tf.nn.bias_add(conv_kernel_2_sub3, paramInitWeight['B5_C2_B'])
layer_2_sub3 = tf.nn.relu(bias_layer_2_sub3)
conv_kernel_3_sub3 = tf.nn.conv2d(layer_2_sub3, paramInitWeight['B5_C3_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_3_sub3 = tf.nn.bias_add(conv_kernel_3_sub3, paramInitWeight['B5_C3_B'])
layer_3_sub3 = tf.nn.relu(bias_layer_3_sub3)
last_pool_sub3 = tf.layers.max_pooling2d(inputs=layer_3_sub3, pool_size=[2, 2], strides=2)
last_flattening_sub3 = tf.reshape(last_pool_sub3, [-1, 7*7*512])
extractF_sub3 = tf.nn.relu_layer(last_flattening_sub3, paramInitWeight['fc1_K'], paramInitWeight['fc1_B']) 

# extract sub4
conv_kernel_1_sub4 = tf.nn.conv2d(xTrainDataSub4, paramInitWeight['B5_C1_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_1_sub4 = tf.nn.bias_add(conv_kernel_1_sub4, paramInitWeight['B5_C1_B'])
layer_1_sub4 = tf.nn.relu(bias_layer_1_sub4)
conv_kernel_2_sub4 = tf.nn.conv2d(layer_1_sub4, paramInitWeight['B5_C2_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_2_sub4 = tf.nn.bias_add(conv_kernel_2_sub4, paramInitWeight['B5_C2_B'])
layer_2_sub4 = tf.nn.relu(bias_layer_2_sub4)
conv_kernel_3_sub4 = tf.nn.conv2d(layer_2_sub4, paramInitWeight['B5_C3_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_3_sub4 = tf.nn.bias_add(conv_kernel_3_sub4, paramInitWeight['B5_C3_B'])
layer_3_sub4 = tf.nn.relu(bias_layer_3_sub4)
last_pool_sub4 = tf.layers.max_pooling2d(inputs=layer_3_sub4, pool_size=[2, 2], strides=2)
last_flattening_sub4 = tf.reshape(last_pool_sub4, [-1, 7*7*512])
extractF_sub4 = tf.nn.relu_layer(last_flattening_sub4, paramInitWeight['fc1_K'], paramInitWeight['fc1_B']) 

# extract sub5
conv_kernel_1_sub5 = tf.nn.conv2d(xTrainDataSub5, paramInitWeight['B5_C1_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_1_sub5 = tf.nn.bias_add(conv_kernel_1_sub5, paramInitWeight['B5_C1_B'])
layer_1_sub5 = tf.nn.relu(bias_layer_1_sub5)
conv_kernel_2_sub5 = tf.nn.conv2d(layer_1_sub5, paramInitWeight['B5_C2_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_2_sub5 = tf.nn.bias_add(conv_kernel_2_sub5, paramInitWeight['B5_C2_B'])
layer_2_sub5 = tf.nn.relu(bias_layer_2_sub5)
conv_kernel_3_sub5 = tf.nn.conv2d(layer_2_sub5, paramInitWeight['B5_C3_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_3_sub5 = tf.nn.bias_add(conv_kernel_3_sub5, paramInitWeight['B5_C3_B'])
layer_3_sub5 = tf.nn.relu(bias_layer_3_sub5)
last_pool_sub5 = tf.layers.max_pooling2d(inputs=layer_3_sub5, pool_size=[2, 2], strides=2)
last_flattening_sub5 = tf.reshape(last_pool_sub5, [-1, 7*7*512])
extractF_sub5 = tf.nn.relu_layer(last_flattening_sub5, paramInitWeight['fc1_K'], paramInitWeight['fc1_B']) 

# extract sub6
conv_kernel_1_sub6 = tf.nn.conv2d(xTrainDataSub6, paramInitWeight['B5_C1_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_1_sub6 = tf.nn.bias_add(conv_kernel_1_sub6, paramInitWeight['B5_C1_B'])
layer_1_sub6 = tf.nn.relu(bias_layer_1_sub6)
conv_kernel_2_sub6 = tf.nn.conv2d(layer_1_sub6, paramInitWeight['B5_C2_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_2_sub6 = tf.nn.bias_add(conv_kernel_2_sub6, paramInitWeight['B5_C2_B'])
layer_2_sub6 = tf.nn.relu(bias_layer_2_sub6)
conv_kernel_3_sub6 = tf.nn.conv2d(layer_2_sub6, paramInitWeight['B5_C3_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_3_sub6 = tf.nn.bias_add(conv_kernel_3_sub6, paramInitWeight['B5_C3_B'])
layer_3_sub6 = tf.nn.relu(bias_layer_3_sub6)
last_pool_sub6 = tf.layers.max_pooling2d(inputs=layer_3_sub6, pool_size=[2, 2], strides=2)
last_flattening_sub6 = tf.reshape(last_pool_sub6, [-1, 7*7*512])
extractF_sub6 = tf.nn.relu_layer(last_flattening_sub6, paramInitWeight['fc1_K'], paramInitWeight['fc1_B']) 

# extract sub7
conv_kernel_1_sub7 = tf.nn.conv2d(xTrainDataSub7, paramInitWeight['B5_C1_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_1_sub7 = tf.nn.bias_add(conv_kernel_1_sub7, paramInitWeight['B5_C1_B'])
layer_1_sub7 = tf.nn.relu(bias_layer_1_sub7)
conv_kernel_2_sub7 = tf.nn.conv2d(layer_1_sub7, paramInitWeight['B5_C2_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_2_sub7 = tf.nn.bias_add(conv_kernel_2_sub7, paramInitWeight['B5_C2_B'])
layer_2_sub7 = tf.nn.relu(bias_layer_2_sub7)
conv_kernel_3_sub7 = tf.nn.conv2d(layer_2_sub7, paramInitWeight['B5_C3_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_3_sub7 = tf.nn.bias_add(conv_kernel_3_sub7, paramInitWeight['B5_C3_B'])
layer_3_sub7 = tf.nn.relu(bias_layer_3_sub7)
last_pool_sub7 = tf.layers.max_pooling2d(inputs=layer_3_sub7, pool_size=[2, 2], strides=2)
last_flattening_sub7 = tf.reshape(last_pool_sub7, [-1, 7*7*512])
extractF_sub7 = tf.nn.relu_layer(last_flattening_sub7, paramInitWeight['fc1_K'], paramInitWeight['fc1_B']) 

# extract sub8
conv_kernel_1_sub8 = tf.nn.conv2d(xTrainDataSub8, paramInitWeight['B5_C1_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_1_sub8 = tf.nn.bias_add(conv_kernel_1_sub8, paramInitWeight['B5_C1_B'])
layer_1_sub8 = tf.nn.relu(bias_layer_1_sub8)
conv_kernel_2_sub8 = tf.nn.conv2d(layer_1_sub8, paramInitWeight['B5_C2_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_2_sub8 = tf.nn.bias_add(conv_kernel_2_sub8, paramInitWeight['B5_C2_B'])
layer_2_sub8 = tf.nn.relu(bias_layer_2_sub8)
conv_kernel_3_sub8 = tf.nn.conv2d(layer_2_sub8, paramInitWeight['B5_C3_K'], [1, 1, 1, 1], padding='SAME')
bias_layer_3_sub8 = tf.nn.bias_add(conv_kernel_3_sub8, paramInitWeight['B5_C3_B'])
layer_3_sub8 = tf.nn.relu(bias_layer_3_sub8)
last_pool_sub8 = tf.layers.max_pooling2d(inputs=layer_3_sub8, pool_size=[2, 2], strides=2)
last_flattening_sub8 = tf.reshape(last_pool_sub8, [-1, 7*7*512])
extractF_sub8 = tf.nn.relu_layer(last_flattening_sub8, paramInitWeight['fc1_K'], paramInitWeight['fc1_B']) 

x_ext = tf.concat([extractF_sub1,extractF_sub2,extractF_sub3,extractF_sub4,extractF_sub5,extractF_sub6,extractF_sub7,extractF_sub8], 1)

dense1 = tf.layers.dense(x_ext, 1000, activation = tf.nn.relu)
dense2 = tf.layers.dense(dense1,500,activation = tf.nn.relu)
yHat = tf.layers.dense(dense1,1,activation = None)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=yHat))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.00001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

y_pred = tf.to_float(tf.greater(tf.sigmoid(yHat),0.5))
accuracy = tf.reduce_mean(tf.to_float(tf.equal(y,y_pred)))

m = trainData.shape[0]
minibatch_size = 480
num_epochs = 1000
display_step = 1
costs = []
trainCosts = []
devCosts = []
trainAccuracies = []
devAccuracies = []

with tf.Session() as sess:
  # initialize all variables
  sess.run(tf.global_variables_initializer())

  for epoch in range(num_epochs):
    cost_in_each_epoch = 0
    num_minibatches = int(m / minibatch_size)
    seed = seed + 1
    minibatches = random_mini_batches4CNN(trainData, trainLabels, minibatch_size, seed)

    for minibatch in minibatches:
      (minibatch_X, minibatch_Y) = minibatch

      # let's start training
      _, c = sess.run([optimizer, cost], feed_dict={x: minibatch_X, y: minibatch_Y})
      cost_in_each_epoch += c
      
    # you can uncomment next two lines of code for printing cost when training
    if (epoch+1) % display_step == 0:
      costs.append(cost_in_each_epoch)

      # yHatDev = sess.run(yHat, feed_dict={x: testData})
      trainC = sess.run(cost, feed_dict={x: trainData, y: trainLabels})
      trainCosts.append(trainC)

      devC = sess.run(cost, feed_dict={x: testData, y: testLabels})
      devCosts.append(devC)
      
      trainAccuracy = accuracy.eval({x: trainData, y: trainLabels})
      trainAccuracies.append(trainAccuracy)
      devAccuracy = accuracy.eval({x: testData, y: testLabels})
      devAccuracies.append(devAccuracy)

    if (epoch+1) % (display_step) == 0:
      print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch), "Train cost={}".format(trainC), "Dev cost={}".format(devC), "Train Acc={}".format(trainAccuracy), "Dev Acc={}".format(devAccuracy))

  print("Optimization Finished!")

  # Test model
  print("Train Accuracy:", accuracy.eval({x: trainData, y: trainLabels}))
  print("Test Accuracy:", accuracy.eval({x: testData, y: testLabels}))

  y_predn = sess.run(y_pred, feed_dict={x: testData})
  y_predn = np.squeeze(y_predn)
  print(y_predn.shape)

  print("Precision", precision_score(testLabels, y_predn))
  print("Recall", recall_score(testLabels, y_predn))
  print("f1_score", f1_score(testLabels, y_predn))

  print ("[STATUS] finish time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

  #plot the cost
  plt.figure(0)
  plt.plot(np.squeeze(trainCosts),'b')
  plt.plot(np.squeeze(devCosts),'r')
  plt.ylabel('cost')
  plt.xlabel('iterations (per tens)')
  plt.show()

  plt.figure(1)
  plt.plot(np.squeeze(trainAccuracies),'b')
  plt.plot(np.squeeze(devAccuracies),'r')
  plt.ylabel('accuracy')
  plt.xlabel('iterations (per tens)')
  plt.show()


