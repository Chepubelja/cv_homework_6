import math
import os
import pickle
import pandas as pd

import warnings

from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

from keras.losses import sparse_categorical_crossentropy, kullback_leibler_divergence
from keras.callbacks import EarlyStopping, LearningRateScheduler

from models import *
from optimizers import *
from configs import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#  ===========================================================

def load_cifar_10(path_to_train, path_to_test):
	train_images, train_labels = pickle.load(open(path_to_train, 'rb'))
	test_images, test_labels = pickle.load(open(path_to_test, 'rb'))
	return (train_images, train_labels), (test_images, test_labels)


def save_history_to_file(model_history, exp_name):
	# convert the history.history dict to a pandas DataFrame:
	hist_df = pd.DataFrame(model_history)
	with open(os.path.join(PATH_TO_EXPERIMENTS, '{}.json'.format(exp_name)), 'w') as f:
		hist_df.to_json(f)
	print('Training and validation history are saved to {}.json'.format(exp_name))


if __name__ == '__main__':

	train, test = load_cifar_10(PATH_TO_TRAIN, PATH_TO_TEST)

	train_images, train_labels = train
	test_images, test_labels = test

	import numpy as np
	print(train_images.shape)
	print(test_images.shape)

	# Determine shape of the data
	img_width, img_height = train_images.shape[1], train_images.shape[2]
	num_channels = train_images.shape[3]
	input_shape = (img_width, img_height, num_channels)

	# Parse numbers as floats
	train_images = train_images.astype('float32')
	test_images = test_images.astype('float32')

	# Normalize data
	train_images = train_images / 255
	test_images = test_images / 255

	# Mean image subtraction
	X_train_mean = np.mean(train_images)
	train_images -= X_train_mean
	X_test_mean = np.mean(test_images)
	test_images -= X_test_mean

	# Model configuration
	batch_size = 128

	# Loss function
	loss_function = sparse_categorical_crossentropy

	# LR Scheduler
	def lr_step_decay(epoch):
		drop_rate = 0.5
		epochs_drop = 15
		return 0.001 * math.pow(drop_rate, math.floor(epoch / epochs_drop))


	lr_scheduler = LearningRateScheduler(lr_step_decay)

	num_classes = 10
	num_epochs = 150
	optimizer = adam
	validation_split = 0.2
	verbosity = 1

	X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels,
	                                                  test_size=validation_split, random_state=42)

	# Create the model
	# model = baseline_model(input_shape, num_classes)
	model = AlexNet(input_shape, num_classes)
	# depth = 2 * 9 + 2
	# model = resnet_v2(input_shape=input_shape, depth=depth)

	# Compile the model
	model.compile(loss=loss_function,
	              optimizer=optimizer,
	              metrics=['accuracy'])

	# Adding Early Stopping
	num_of_epochs_no_improv = 10
	early_stopping = EarlyStopping(monitor='val_acc',
	                               patience=num_of_epochs_no_improv,
	                               mode='max',
	                               verbose=1)

	# Data augmentation
	width_shift = 3 / 32
	height_shift = 3 / 32
	flip = True

	datagen = ImageDataGenerator(
		horizontal_flip=flip,
		width_shift_range=width_shift,
		height_shift_range=height_shift,
		data_format='channels_last',
		validation_split=validation_split
	)



	# Fit data to model
	history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
	                              epochs=num_epochs,
	                              validation_data=(X_val, y_val),
	                              steps_per_epoch=len(X_train) // batch_size,
	                              # validation_split=validation_split,
	                              verbose=verbosity,
	                              callbacks=[lr_scheduler, early_stopping])
	# history = model.fit(train_images, train_labels,
	# 						batch_size=batch_size,
	# 						epochs=num_epochs,
	# 						verbose=verbosity,
	# 						validation_split=validation_split)
	                        # callbacks=[early_stopping])

	# Saving history to .json for further visualizations
	save_history_to_file(history.history, EXP_NAME)

	# Generate generalization metrics
	score = model.evaluate(test_images, test_labels, verbose=0)
	print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
