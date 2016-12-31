'''
Train a neural network to recognize traffic signs
Use German Traffic Signs Dataset for training data, and test set data
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim  # TensorFlow-Slim
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import math
import os
import time
import pickle

# Settings/parameters to be used later

# Constants
IMG_SIZE = 32  # square image of size IMG_SIZE x IMG_SIZE
GRAYSCALE = False  # convert image to grayscale?
NUM_CHANNELS = 1 if GRAYSCALE else 3
NUM_CLASSES = 43

# Model parameters
LR = 5e-3  # learning rate
KEEP_PROB = 0.5  # dropout keep probability
OPT = tf.train.GradientDescentOptimizer(learning_rate=LR)  # choose which optimizer to use

# Training process
RESUME = False  # resume training from previously trained model?
NUM_EPOCH = 40
BATCH_SIZE = 128  # batch size for training (relatively small)
BATCH_SIZE_INF = 2048  # batch size for running inference, e.g. calculating accuracy
VALIDATION_SIZE = 0.2  # fraction of total training set to use as validation set
SAVE_MODEL = True  # save trained model to disk?
MODEL_SAVE_PATH = './model.ckpt'  # where to save trained model

########################################################
# Helper functions and generators
########################################################
def rgb_to_gray(images):
	"""
	Convert batch of RGB images to grayscale
	Use simple average of R, G, B values, not weighted average

	Arguments:
		* Batch of RGB images, tensor of shape (batch_size, 32, 32, 3)

	Returns:
		* Batch of grayscale images, tensor of shape (batch_size, 32, 32, 1)
	"""
	images_gray = np.average(images, axis=3)
	images_gray = np.expand_dims(images_gray, axis=3)
	return images_gray


def preprocess_data(X, y):
	"""
	Preprocess image data, and convert labels into one-hot

	Arguments:
		* X: Array of images
		* y: Array of labels

	Returns:
		* Preprocessed X, one-hot version of y
	"""
	# Convert from RGB to grayscale if applicable
	if GRAYSCALE:
		X = rgb_to_gray(X)

	# Make all image array values fall within the range -1 to 1
	# Note all values in original images are between 0 and 255, as uint8
	X = X.astype('float32')
	X = (X - 128.) / 128.

	# Convert the labels from numerical labels to one-hot encoded labels
	y_onehot = np.zeros((y.shape[0], NUM_CLASSES))
	for i, onehot_label in enumerate(y_onehot):
		onehot_label[y[i]] = 1.
	y = y_onehot

	return X, y


def next_batch(X, y, batch_size, augment_data):
	"""
	Generator to generate data and labels
	Each batch yielded is unique, until all data is exhausted
	If all data is exhausted, the next call to this generator will throw a StopIteration

	Arguments:
		* X: image data, a tensor of shape (dataset_size, 32, 32, 3)
		* y: labels, a tensor of shape (dataset_size,)  <-- i.e. a list
		* batch_size: Size of the batch to yield
		* augment_data: Boolean value, whether to augment the data (i.e. perform image transform)

	Yields:
		A tuple of (images, labels), where:
			* images is a tensor of shape (batch_size, 32, 32, 3)
			* labels is a tensor of shape (batch_size,)
	"""
	# A generator in this case is likely overkill,
	# but using a generator is a more scalable practice,
	# since future datasets may be too large to fit in memory

	# We know X and y are randomized from the train/validation split already,
	# so just sequentially yield the batches
	start_idx = 0
	while start_idx < X.shape[0]:
		images = X[start_idx : start_idx + batch_size]
		labels = y[start_idx : start_idx + batch_size]

		yield (np.array(images), np.array(labels))

		start_idx += batch_size


def calculate_accuracy(data_gen, data_size, batch_size, accuracy, x, y, keep_prob, sess):
	"""
	Helper function to calculate accuracy on a particular dataset

	Arguments:
		* data_gen: Generator to generate batches of data
		* data_size: Total size of the data set, must be consistent with generator
		* batch_size: Batch size, must be consistent with generator
		* accuracy, x, y, keep_prob: Tensor objects in the neural network
		* sess: TensorFlow session object containing the neural network graph

	Returns:
		* Float representing accuracy on the data set
	"""
	num_batches = math.ceil(data_size / batch_size)
	last_batch_size = data_size % batch_size

	accs = []  # accuracy for each batch

	for _ in range(num_batches):
		images, labels = next(data_gen)

		# Perform forward pass and calculate accuracy
		# Note we set keep_prob to 1.0, since we are performing inference
		acc = sess.run(accuracy, feed_dict={x: images, y: labels, keep_prob: 1.})
		accs.append(acc)

	# Calculate average accuracy of all full batches (the last batch is the only partial batch)
	acc_full = np.mean(accs[:-1])

	# Calculate weighted average of accuracy accross batches
	acc = (acc_full * (data_size - last_batch_size) + accs[-1] * last_batch_size) / data_size

	return acc


########################################################
# Neural network architecture
########################################################
def neural_network():
	"""
	Define neural network architecture
	Return relevant tensor references
	"""
	with tf.variable_scope('neural_network'):
		# Tensors representing input images and labels
		x = tf.placeholder('float', [None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])
		y = tf.placeholder('float', [None, NUM_CLASSES])

		# Placeholder for dropout keep probability
		keep_prob = tf.placeholder(tf.float32)

		# Neural network architecture: Convolutional Neural Network (CNN)
		# Using TensorFlow-Slim to build the network:
		# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim

		# Use batch normalization for all convolution layers
		with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
			# Given x shape is (32, 32, 3)
			# Conv and pool layers
			net = slim.conv2d(x, 16, [3, 3], scope='conv0')  # output shape: (32, 32, 16)
			net = slim.max_pool2d(net, [3, 3], 1, padding='SAME', scope='pool0')  # output shape: (32, 32, 16)
			net = slim.conv2d(net, 64, [5, 5], 3, padding='VALID', scope='conv1')  # output shape: (10, 10, 64)
			net = slim.max_pool2d(net, [3, 3], 1, scope='pool1')  # output shape: (8, 8, 64)
			net = slim.conv2d(net, 128, [3, 3], scope='conv2')  # output shape: (8, 8, 128)
			net = slim.conv2d(net, 64, [3, 3], scope='conv3')  # output shape: (8, 8, 64)
			net = slim.max_pool2d(net, [3, 3], 1, scope='pool3')  # output shape: (6, 6, 64)

			# Final fully-connected layers
			net = tf.contrib.layers.flatten(net)
			net = slim.fully_connected(net, 1024, scope='fc4')
			net = tf.nn.dropout(net, keep_prob)
			net = slim.fully_connected(net, 1024, scope='fc5')
			net = tf.nn.dropout(net, keep_prob)
			net = slim.fully_connected(net, NUM_CLASSES, scope='fc6')

		# Final output (logits)
		logits = net

		# Loss (data loss and regularization loss) and optimizer
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))
		optimizer = OPT.minimize(loss)

		# Prediction (used during inference)
		predictions = tf.argmax(logits, 1)

		# Accuracy metric calculation
		correct_predictions = tf.equal(predictions, tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

	# Return relevant tensor references
	return x, y, keep_prob, logits, optimizer, predictions, accuracy


########################################################
# Main training function
########################################################
def run_training():
	"""
	Load training and test data
	Run training process
	Plot train/validation accuracies
	Report test accuracy
	Save model
	"""
	########################################################
	# Load training and test data
	########################################################
	training_file = 'train_aug.p'
	testing_file = 'test.p'

	with open(training_file, mode='rb') as f:
		train = pickle.load(f)
	with open(testing_file, mode='rb') as f:
		test = pickle.load(f)

	X_train, y_train = train['features'], train['labels']
	X_test, y_test = test['features'], test['labels']

	# Basic data summary
	n_train = X_train.shape[0]
	n_test = X_test.shape[0]
	image_shape = X_train.shape[1:3]
	n_classes = np.unique(y_train).shape[0]

	########################################################
	# Data pre-processing
	########################################################
	X_train, y_train = preprocess_data(X_train, y_train)
	X_test, y_test = preprocess_data(X_test, y_test)

	########################################################
	# Train/validation split
	########################################################
	X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE)

	# Launch the graph
	with tf.Graph().as_default(), tf.Session() as sess:
		########################################################
		# "Instantiate" neural network, get relevant tensors
		########################################################
		x, y, keep_prob, logits, optimizer, predictions, accuracy = neural_network()

		########################################################
		# Training process
		########################################################
		# TF saver to save/restore trained model
		saver = tf.train.Saver()

		if RESUME:
			print('Restoring previously trained model at %s' % MODEL_SAVE_PATH)
			# Restore previously trained model
			saver.restore(sess, MODEL_SAVE_PATH)

			# Restore previous accuracy history
			with open('accuracy_history.p', 'rb') as f:
				accuracy_history = pickle.load(f)
		else:
			print('Training model from scratch')
			# Variable initialization
			init = tf.initialize_all_variables()
			sess.run(init)

			# For book-keeping, keep track of training and validation accuracy over epochs, like such:
			# [(train_acc_epoch1, valid_acc_epoch1), (train_acc_epoch2, valid_acc_epoch2), ...]
			accuracy_history = []

		# Record time elapsed for performance check
		last_time = time.time()
		train_start_time = time.time()

		# Run NUM_EPOCH epochs of training
		for epoch in range(NUM_EPOCH):
			# Instantiate generator for training data
			train_gen = next_batch(X_train, y_train, BATCH_SIZE, True)

			# How many batches to run per epoch
			num_batches_train = math.ceil(X_train.shape[0] / BATCH_SIZE)

			# Run training on each batch
			for _ in range(num_batches_train):
				# Obtain the training data and labels from generator
				images, labels = next(train_gen)

				# Perform gradient update (i.e. training step) on current batch
				sess.run(optimizer, feed_dict={x: images, y: labels, keep_prob: KEEP_PROB})

			# Calculate training and validation accuracy across the *entire* train/validation set
			# If train/validation size % batch size != 0
			# then we must calculate weighted average of the accuracy of the final (partial) batch,
			# w.r.t. the rest of the full batches

			# Training set
			train_gen = next_batch(X_train, y_train, BATCH_SIZE_INF, True)
			train_size = X_train.shape[0]
			train_acc = calculate_accuracy(train_gen, train_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)

			# Validation set
			valid_gen = next_batch(X_valid, y_valid, BATCH_SIZE_INF, True)
			valid_size = X_valid.shape[0]
			valid_acc = calculate_accuracy(valid_gen, valid_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)

			# Record and report train/validation/test accuracies for this epoch
			accuracy_history.append((train_acc, valid_acc))

			# Print accuracy every 10 epochs
			if (epoch+1) % 10 == 0 or epoch == 0 or (epoch+1) == NUM_EPOCH:
				print('Epoch %d -- Train acc.: %.4f, Validation acc.: %.4f, Elapsed time: %.2f sec' %\
					(epoch+1, train_acc, valid_acc, time.time() - last_time))
				last_time = time.time()

		total_time = time.time() - train_start_time
		print('Total elapsed time: %.2f sec (%.2f min)' % (total_time, total_time/60))

		# After training is complete, evaluate accuracy on test set
		print('Calculating test accuracy...')
		test_gen = next_batch(X_test, y_test, BATCH_SIZE_INF, False)
		test_size = X_test.shape[0]
		test_acc = calculate_accuracy(test_gen, test_size, BATCH_SIZE_INF, accuracy, x, y, keep_prob, sess)
		print('Test acc.: %.4f' % (test_acc,))

		if SAVE_MODEL:
			# Save model to disk
			save_path = saver.save(sess, MODEL_SAVE_PATH)
			print('Trained model saved at: %s' % save_path)

			# Also save accuracy history
			print('Accuracy history saved at accuracy_history.p')
			with open('accuracy_history.p', 'wb') as f:
				pickle.dump(accuracy_history, f)

		# Return final test accuracy and accuracy_history
		return test_acc, accuracy_history


########################################################
# Model inference function
########################################################
def run_inference(image_files):
	"""
	Load trained model and run inference on images

	Arguments:
		* images: Array of images on which to run inference

	Returns:
		* Array of strings, representing the model's predictions
	"""
	# Read image files, resize them, convert to numpy arrays w/ dtype=uint8
	images = []
	for image_file in image_files:
		image = Image.open(image_file)
		image = image.convert('RGB')
		image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
		image = np.array(list(image.getdata()), dtype='uint8')
		image = np.reshape(image, (32, 32, 3))

		images.append(image)
	images = np.array(images, dtype='uint8')

	# Pre-process the image (don't care about label, put dummy labels)
	images, _ = preprocess_data(images, np.array([0 for _ in range(images.shape[0])]))

	with tf.Graph().as_default(), tf.Session() as sess:
		# Instantiate the CNN model
		x, y, keep_prob, logits, optimizer, predictions, accuracy = neural_network()

		# Load trained weights
		saver = tf.train.Saver()
		saver.restore(sess, MODEL_SAVE_PATH)

		# Run inference on CNN to make predictions
		preds = sess.run(predictions, feed_dict={x: images, keep_prob: 1.})

	# Load signnames.csv to map label number to sign string
	label_map = {}
	with open('signnames.csv', 'r') as f:
		first_line = True
		for line in f:
			# Ignore first line
			if first_line:
				first_line = False
				continue

			# Populate label_map
			label_int, label_string = line.split(',')
			label_int = int(label_int)

			label_map[label_int] = label_string

	final_preds = [label_map[pred] for pred in preds]

	return final_preds


if __name__ == '__main__':
	test_acc, accuracy_history = run_training()

	# Obtain list of sample image files
	sample_images = ['sample_images/' + image_file for image_file in os.listdir('sample_images')]
	preds = run_inference(sample_images)
	print('Predictions on sample images:')
	for i in range(len(sample_images)):
		print('%s --> %s' % (sample_images[i], preds[i]))
