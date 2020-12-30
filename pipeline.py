import os
import pickle
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

from tensorflow.keras.losses import sparse_categorical_crossentropy

from models import *
from optimizers import *
from configs import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#  ===========================================================

def load_data(path_to_train, path_to_test):
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

	train, test = load_data(PATH_TO_TRAIN, PATH_TO_TEST)

	train_images, train_labels = train
	test_images, test_labels = test

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

	# Model configuration
	batch_size = 100

	# Loss function
	loss_function = sparse_categorical_crossentropy

	num_classes = 100
	num_epochs = 50
	optimizer = adam
	validation_split = 0.2
	verbosity = 1

	# Create the model
	model = baseline_model(input_shape, num_classes)

	# Compile the model
	model.compile(loss=loss_function,
	              optimizer=optimizer,
	              metrics=['accuracy'])

	# Fit data to model
	history = model.fit(train_images, train_labels,
							batch_size=batch_size,
							epochs=num_epochs,
							verbose=verbosity,
							validation_split=validation_split)

	# Saving history to .json for further visualizations
	save_history_to_file(history.history, EXP_NAME)

	# Generate generalization metrics
	score = model.evaluate(test_images, test_labels, verbose=0)
	print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
