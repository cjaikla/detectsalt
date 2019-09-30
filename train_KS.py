# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
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

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))

x = tf.placeholder(shape=[None, 32768], dtype=tf.float32, name = "x")
y = tf.placeholder(shape=[None, 1], dtype=tf.float32, name = "y")

dense1 = tf.layers.dense(x, 1000, activation = tf.nn.relu)
dense2 = tf.layers.dense(dense1,500,activation = tf.nn.relu)
yHat = tf.layers.dense(dense1,1,activation = None)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=yHat))
optimizer = tf.train.AdamOptimizer(learning_rate= 0.00001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(cost)

y_pred = tf.to_float(tf.greater(tf.sigmoid(yHat),0.5))
accuracy = tf.reduce_mean(tf.to_float(tf.equal(y,y_pred)))

m = trainData.shape[0]
minibatch_size = 450
num_epochs = 10
display_step = 10
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
    minibatches = random_mini_batches(trainData, trainLabels, minibatch_size, seed)

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

  y_predn = sess.run([y_pred], feed_dict={x: testData})
  y_predn = np.squeeze(y_predn)
  print("Precision", precision_score(testLabels, y_predn))
  print("Recall", recall_score(testLabels, y_predn))
  print("f1_score", f1_score(testLabels, y_predn))

  # plot the cost
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

# # use logistic regression as the model
# print ("[INFO] creating model...")
# model = LogisticRegression(random_state=seed)
# model.fit(trainData, trainLabels)

# # use rank-1 and rank-5 predictions
# print ("[INFO] evaluating model...")
# f = open(results, "w")
# rank_1 = 0
# rank_5 = 0

# # loop over test data
# for (label, features) in zip(testLabels, testData):
#   # predict the probability of each class label and
#   # take the top-5 class labels
#   predictions = model.predict_proba(np.atleast_2d(features))[0]
#   predictions = np.argsort(predictions)[::-1][:5]

#   # rank-1 prediction increment
#   if label == predictions[0]:
#     rank_1 += 1

#   # rank-5 prediction increment
#   if label in predictions:
#     rank_5 += 1

# # convert accuracies to percentages
# rank_1 = (rank_1 / float(len(testLabels))) * 100
# rank_5 = (rank_5 / float(len(testLabels))) * 100

# # write the accuracies to file
# f.write("Rank-1: {:.2f}%\n".format(rank_1))
# f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# # evaluate the model of test data
# preds = model.predict(testData)

# # write the classification report to file
# f.write("{}\n".format(classification_report(testLabels, preds)))
# f.close()

# # dump classifier to file
# print ("[INFO] saving model...")
# pickle.dump(model, open(classifier_path, 'wb'))

# # display the confusion matrix
# print ("[INFO] confusion matrix")

# # get the list of training lables
# labels = sorted(list(os.listdir(train_path)))

# # plot the confusion matrix
# cm = confusion_matrix(testLabels, preds)
# sns.heatmap(cm,
#             annot=True,
#             cmap="Set2")
# plt.show()