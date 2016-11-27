'''
Augment the data
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import pickle

# Image data augmentation parameters
ANGLE = 15
TRANSLATION = 0.2
WARP = 0.0  #0.05
#NUM_NEW_IMAGES = 100000
NUM_NEW_IMAGES = 1000

########################################################
# Helper functions
########################################################
def transform_image(image, angle, translation, warp):
	"""
	Transform the image for data augmentation
	
	Arguments:
		* image: Input image
		* angle: Max rotation angle, in degrees. Direction of rotation is random.
		* translation: Max translation amount in both x and y directions,
			expressed as fraction of total image width/height
		* warp: Max warp amount for each of the 3 reference points,
			expressed as fraction of total image width/height
		
	Returns:
		* Transformed image as an np.array() object
	"""
	height, width, channels = image.shape
	
	# Rotation
	center = (width//2, height//2)
	angle_rand = np.random.uniform(-angle, angle)
	rotation_mat = cv2.getRotationMatrix2D(center, angle_rand, 1)
	
	image = cv2.warpAffine(image, rotation_mat, (width, height))
	
	# Translation
	x_offset = translation * width * np.random.uniform(-1, 1)
	y_offset = translation * height * np.random.uniform(-1, 1)
	translation_mat = np.array([[1, 0, x_offset], [0, 1, y_offset]])
	
	image = cv2.warpAffine(image, translation_mat, (width, height))
	
	# Warp
	# NOTE: The commented code below is left for reference
	# The warp function tends to blur the image, so it is not useds
	'''
	src_triangle = np.float32([[0, 0], [0, height], [width, 0]])
	x_offsets = [warp * width * np.random.uniform(-1, 1) for _ in range(3)]
	y_offsets = [warp * height * np.random.uniform(-1, 1) for _ in range(3)]
	dst_triangle = np.float32([[x_offsets[0], y_offsets[0]],\
							 [x_offsets[1], height + y_offsets[1]],\
							 [width + x_offsets[2], y_offsets[2]]])
	warp_mat = cv2.getAffineTransform(src_triangle, dst_triangle)
	
	image = cv2.warpAffine(image, warp_mat, (width, height))
	'''
	
	return image


def display_random_images(images):
	"""
	Display random image, and transformed versions of it
	For debug only
	"""
	image = images[np.random.randint(images.shape[0])]

	# Show original image for reference
	plt.subplot(3, 3, 1)
	plt.imshow(image)
	plt.title('Original Image')

	for i in range(8):
		image_x = transform_image(image, ANGLE, TRANSLATION, WARP)
		plt.subplot(3, 3, i+2)
		plt.imshow(image_x)
		plt.title('Transformed Image %d' % (i+1,))

	plt.tight_layout()
	plt.show()


def display_random_aug(aug_file):
	"""
	Display random images from augmented dataset
	For debug only
	"""
	with open(aug_file, mode='rb') as f:
		aug_data = pickle.load(f)
	images = aug_data['features']

	for i in range(9):
		rand_idx = np.random.randint(images.shape[0])
		image = images[rand_idx]
		plt.subplot(3, 3, i+1)
		plt.imshow(image)
		plt.title('Image Idx: %d' % (rand_idx,))

	plt.tight_layout()
	plt.show()


########################################################
# Main function
########################################################
def data_aug(orig_file, new_file, num_new_images):
	"""
	blah
	"""
	# Load original dataset
	with open(orig_file, mode='rb') as f:
		orig_data = pickle.load(f)

	orig_X, orig_y = orig_data['features'], orig_data['labels']

	# Create NUM_NEW_IMAGES new images, via image transform on random original image
	for i in range(NUM_NEW_IMAGES):
		# Pick a random image from original dataset to transform
		rand_idx = np.random.randint(orig_X.shape[0])

		# Create new image
		image = transform_image(orig_X[rand_idx], ANGLE, TRANSLATION, WARP)

		# Add new data to augmented dataset
		if i == 0:
			new_X = np.expand_dims(image, axis=0)
			new_y = np.array([orig_y[rand_idx]])
		else:
			new_X = np.concatenate((new_X, np.expand_dims(image, axis=0)))
			new_y = np.append(new_y, orig_y[rand_idx])

		if (i+1) % 1000 == 0:
			print('%d new images generated' % (i+1,))

	new_X = np.concatenate((orig_X, new_X))
	new_y = np.concatenate((orig_y, new_y))

	# Create dict of new data, and write it to disk via pickle file
	new_data = {'features': new_X, 'labels': new_y}
	with open(new_file, mode='wb') as f:
		pickle.dump(new_data, f)

	return new_data


if __name__ == '__main__':
	# This part is for visualization and/or debug
	#with open('train.p', mode='rb') as f:
	#	orig_data = pickle.load(f)
	#display_random_images(orig_data['features'])

	# This actually creates the augmented dataset
	data_aug('train.p', 'train_aug.p', NUM_NEW_IMAGES)
	
	# For debug, display random images from augmented dataset
	#display_random_aug('train_aug.p')

