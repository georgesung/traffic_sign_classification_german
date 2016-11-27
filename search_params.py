'''
Searching for the optimal parameter combinations.
'''
import tensorflow as tf
import tensorflow.contrib.slim as slim  # TensorFlow-Slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
import math
import os
import time
import pickle

import traffic_signs

# Configure the parameter search
NUM_ITERS = 20  # how many iterations of random parameter search
REPORT_FILE = 'search_params.csv'  # save results to this file
RESUME = False  # resume search from previous run?

# Lists of parameter values on which to perform random search (w/o replacement)
LR = [1e-2, 5e-3, 1e-3, 5e-4]
REG_SCALE = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
KEEP_PROB = [1., 0.85, 0.65, 0.5]

# Helper functions
def choose_params(results):
	'''
	Choose random parameter combination w/o replacement
	results is the dict that maps (param1, param2, ...) --> score
	Returns tuple of chosen parameters
	'''
	# Keep trying random parameter combinations until we get a unique combination
	while True:
		lr = np.random.choice(LR)
		reg_scale = np.random.choice(REG_SCALE)
		keep_prob = np.random.choice(KEEP_PROB)

		params = (lr, reg_scale, keep_prob)

		if params not in results:
			break

	return params

def set_params(params):
	'''
	Sets the parameters specified in params tuple
	'''
	traffic_signs.LR = params[0]
	traffic_signs.REG_SCALE = params[1]
	traffic_signs.KEEP_PROB = params[2]


##########################################
# Main script to perform parameter search
##########################################

# Dictionary to store results
# (param1, param2, ...) --> score
if RESUME:
	with open('search_params_result.p', 'rb') as f:
		results = pickle.load(f)

	with open('search_params_acc_hist.p', 'rb') as f:
		acc_hist = pickle.load(f)

	# Assume the csv file is populated already, w/ previous results
else:
	results = {}
	acc_hist = {}

	# Write csv header of report file
	with open(REPORT_FILE, 'w') as f:
		f.write('LR,REG_SCALE,KEEP_PROB,final_test_acc\n')

for _ in range(NUM_ITERS):
	params = choose_params(results)  # choose random parameter combination w/o replacement
	set_params(params)  # set the parameters

	print('Using params: %f, %f, %f' % (*params,))

	# Run training
	test_acc, accuracy_history = traffic_signs.run_training()

	# Record results, append results to report file, and dump results dict to pickle file
	results[params] = test_acc
	with open(REPORT_FILE, 'a') as f:
		f.write('%f,%f,%f,%f\n' % (*params, test_acc))
	with open('search_params_result.p', 'wb') as f:
		pickle.dump(results, f)

	# For possible debug, save the entire accuracy history to pickle file
	acc_hist[params] = accuracy_history
	with open('search_params_acc_hist.p', 'wb') as f:
		pickle.dump(acc_hist, f)
