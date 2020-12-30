from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization


def baseline_model(input_shape, num_classes):
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	return model


def AlexNet(input_shape, num_classes):
	model = Sequential()
	model.add(Conv2D(48, kernel_size=(11, 11),
	                 activation='relu',
	                 input_shape=input_shape, padding="same"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding="same"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(192, (3, 3), activation='relu', padding="same"))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(192, (3, 3), activation='relu', padding="same"))
	model.add(BatchNormalization())
	model.add(Conv2D(192, (3, 3), activation='relu', padding="same"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(768, activation='relu'))
	model.add(Dense(768, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	return model
